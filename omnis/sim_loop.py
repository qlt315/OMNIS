"""Shared discrete-task simulation loop for OMNIS+ and baselines."""

from __future__ import annotations

import time

import numpy as np

from omnis.assoc_info import update_cell_compute_state
from omnis.bcd_loop import run_bcd_slot


def _is_greedy_mab(agent):
    """GDO updates a per-branch UCB from completion accuracy.

    RL agents also implement ``learn_after_slot``, but they stash a slot
    bundle (``team_r`` / ``next_global``) through ``_flush_pending_learn``.
    """
    return (
        hasattr(agent, "learn_after_slot")
        and not hasattr(agent, "_flush_pending_learn")
        and not hasattr(agent, "_mab_allow_update")
        and not hasattr(agent, "optimizers")
        and not hasattr(agent, "update_gp")
    )


def flush_pending_learn(agent):
    pend = getattr(agent, "_pending_learn", None)
    if pend is None:
        agent._last_parallel_update_s = 0.0
        return
    # RL baselines stash slot_info (team_r / dpp_targets), not MAB Acc payloads.
    if isinstance(pend, dict) and (
            "team_r" in pend or "dpp_targets" in pend
            or not ("acc" in pend or "reward" in pend)):
        if hasattr(agent, "_flush_pending_learn"):
            agent._flush_pending_learn()
        elif hasattr(agent, "learn_after_slot"):
            t0 = time.time()
            agent.learn_after_slot(pend)
            agent.update_time = float(getattr(agent, "update_time", 0.0)) + (
                time.time() - t0)
            agent._pending_learn = None
            agent._last_parallel_update_s = time.time() - t0
        else:
            agent._pending_learn = None
            agent._last_parallel_update_s = 0.0
        return
    # Greedy learner (GDO): per-branch UCB update, no GP bandit hook.
    if _is_greedy_mab(agent):
        t0 = time.time()
        agent.learn_after_slot(pend)
        agent.update_time = float(getattr(agent, "update_time", 0.0)) + (
            time.time() - t0)
        agent._pending_learn = None
        agent._last_parallel_update_s = time.time() - t0
        return
    apply_learn_payload(agent, pend)
    agent._pending_learn = None


def apply_learn_payload(agent, pend):
    """Register Acc/reward for the subset of users in ``pend``."""
    users = pend.get("users") or list(pend["acc"].keys())
    if not users:
        agent._last_parallel_update_s = 0.0
        return
    if not hasattr(agent, "_mab_allow_update"):
        agent._last_parallel_update_s = 0.0
        return
    if pend.get("algo") == "causal" and hasattr(agent, "causal_mab"):
        records = [(u, pend["mcs"][u], pend["acc"][u]) for u in users]
        allow = agent._mab_allow_update()
        add_times = []
        for user, mcs_realized, acc_obs in records:
            t_u = time.time()
            agent.causal_mab.register_outcome(
                user, mcs_realized, acc_obs, do_update=allow)
            add_times.append(time.time() - t_u)
        t_trim = time.time()
        if allow:
            if agent.causal_mab.shared:
                agent.causal_mab.gp.maybe_trim()
            else:
                for gp in agent.causal_mab._gps.values():
                    gp.maybe_trim()
        trim_s = time.time() - t_trim
        agent._mab_update_slots += 1
        if getattr(agent, "centralized", False):
            # One controller applies every label, then trims the shared GP.
            agent._last_parallel_update_s = float(sum(add_times) + trim_s)
        else:
            agent._last_parallel_update_s = max(add_times) if add_times else 0.0
    elif hasattr(agent, "optimizers"):
        allow = agent._mab_allow_update()
        user_times = []
        for user in users:
            t_u = time.time()
            optimizer_m = agent.optimizers[user]
            model_m = pend["model_selection"][user]["model"]
            action_m = next(
                (i for i, m in enumerate(agent.models) if m["name"] == model_m),
                None,
            )
            action_dic_m = {
                "model": action_m,
                "cell_rank": pend["model_selection"][user].get("cell_rank", 0),
            }
            if allow:
                optimizer_m.register(
                    pend["context"][user], action_dic_m, pend["reward"][user])
            user_times.append(time.time() - t_u)
        agent._mab_update_slots += 1
        agent._last_parallel_update_s = max(user_times) if user_times else 0.0
    elif hasattr(agent, "update_gp"):
        # Legacy joint reward-GP (not centralized causal): flat context/action.
        allow = agent._mab_allow_update()
        t0 = time.time()
        if allow:
            flat_ctx = {}
            flat_act = {}
            for user in agent.users:
                ctx = pend["context"].get(user)
                if ctx is None:
                    ctx = {
                        "delay_constraint": 0.0,
                        "energy_constraint": 0.0,
                        "transmission_rate": 0.0,
                        "energy_weight": 0.0,
                        "delay_weight": 1.0,
                    }
                flat_ctx[f"{user}_delay_constraint"] = ctx.get(
                    "delay_constraint", 0.0)
                flat_ctx[f"{user}_energy_constraint"] = ctx.get(
                    "energy_constraint", 0.0)
                flat_ctx[f"{user}_transmission_rate"] = ctx.get(
                    "transmission_rate", 0.0)
                flat_ctx[f"{user}_energy_weight"] = ctx.get(
                    "energy_weight", 0.0)
                flat_ctx[f"{user}_delay_weight"] = ctx.get(
                    "delay_weight", 1.0)
                ms = pend["model_selection"].get(user, {"model": agent.models[0]["name"], "cell_rank": 0})
                model_idx = next(
                    (i for i, m in enumerate(agent.models)
                     if m["name"] == ms.get("model")),
                    0,
                )
                flat_act[f"{user}_model"] = model_idx
                flat_act[f"{user}_cell_rank"] = int(ms.get("cell_rank", 0))
            agent.update_gp(flat_ctx, flat_act, pend["reward"])
            agent._mab_update_slots = getattr(agent, "_mab_update_slots", 0) + 1
        agent._last_parallel_update_s = time.time() - t0
    else:
        agent._last_parallel_update_s = 0.0


def _select_models(agent, context_dic, task_dic, cand_cells_dic,
                   sinr_est_db_all_dic, trans_rate_dic, time_slot=0):
    algo = getattr(agent, "algo", None)
    if algo == "causal" and hasattr(agent, "model_selection_causal"):
        return agent.model_selection_causal(
            task_dic, cand_cells_dic, sinr_est_db_all_dic, trans_rate_dic)
    if hasattr(agent, "select_actions"):
        return agent.select_actions(
            cand_cells_dic, task_dic, sinr_est_db_all_dic, time_slot)
    import inspect
    sig = inspect.signature(agent.model_selection)
    n = len(sig.parameters)
    if n <= 1:
        return agent.model_selection(cand_cells_dic)
    return agent.model_selection(
        context_dic, task_dic, cand_cells_dic, sinr_est_db_all_dic)


def run_discrete_simulation(agent):
    """Poisson task arrivals, sticky association, MD/ES queues, BCD per slot."""
    agent._pending_learn = None
    es = agent.es_params
    for t in range(agent.time_slot_num):
        (_snr_best, trans_rate_dic, cand_cells_dic,
         sinr_est_db_all_dic, sinr_true_db_all_dic) = agent.get_trans_rate(t)

        task_dic = agent.generate_tasks(t)
        for user in agent.users:
            draws = task_dic[user].get("qos_draws") or []
            n_arrivals = int(task_dic[user]["n_arrivals"])
            if len(draws) >= n_arrivals and n_arrivals > 0:
                for qos in draws[:n_arrivals]:
                    agent.pipeline.enqueue_arrivals(user, 1, qos)
            else:
                agent.pipeline.enqueue_arrivals(user, n_arrivals, task_dic[user])

        if int(getattr(agent, "learn_delay_slots", 0) or 0) > 0:
            flush_pending_learn(agent)
            if hasattr(agent, "update_time"):
                agent.update_time += float(
                    getattr(agent, "_last_parallel_update_s", 0.0))

        context_dic = agent.observe_context(task_dic, trans_rate_dic)

        # Per-task arm selection: only idle MDs with pending work choose a new arm.
        model_selection_dic = {}
        cell_dic = {}
        for user in agent.users:
            locked_cell = agent.pipeline.locked_cell(user)
            locked_model = agent.pipeline.locked_model(user)
            if locked_cell is not None and locked_model is not None:
                model_selection_dic[user] = {
                    "model": locked_model,
                    "cell_rank": 0,
                }
                cell_dic[user] = locked_cell

        admit_users = [
            u for u in agent.users
            if agent.pipeline.is_idle(u) and agent.pipeline.pending_len(u) > 0
        ]
        # Align HOL QoS into task_dic before scoring
        for u in admit_users:
            hol = agent.pipeline.pending[u][0]
            task_dic[u]["delay_constraint"] = hol.delay_constraint
            task_dic[u]["energy_constraint"] = hol.energy_constraint
            task_dic[u]["delay_weight"] = hol.delay_weight
            task_dic[u]["energy_weight"] = hol.energy_weight

        # Joint CBO / RL keep the full user set (global state); per-user MABs
        # admit-only. RL still decides each slot so the learning cache has CSI.
        joint_cbo = (
            hasattr(agent, "optimizer")
            and not hasattr(agent, "optimizers")
        )
        joint_rl = hasattr(agent, "select_actions")
        if admit_users or joint_rl:
            if joint_cbo or joint_rl:
                context_a = agent.observe_context(task_dic, trans_rate_dic)
                sel, cells = _select_models(
                    agent, context_a, task_dic, cand_cells_dic,
                    sinr_est_db_all_dic, trans_rate_dic, time_slot=t)
                for u in admit_users:
                    model_selection_dic[u] = sel[u]
                    cell_dic[u] = cells[u]
            elif admit_users:
                saved_users = agent.users
                agent.users = admit_users
                try:
                    context_a = {u: context_dic[u] for u in admit_users}
                    task_adm = {u: dict(task_dic[u]) for u in admit_users}
                    cand_a = {u: cand_cells_dic[u] for u in admit_users}
                    sinr_a = {u: sinr_est_db_all_dic[u] for u in admit_users}
                    sel, cells = _select_models(
                        agent, context_a, task_adm, cand_a, sinr_a,
                        trans_rate_dic, time_slot=t)
                    model_selection_dic.update(sel)
                    cell_dic.update(cells)
                finally:
                    agent.users = saved_users
            if hasattr(agent, "decision_time"):
                agent.decision_time += float(
                    getattr(agent, "_last_parallel_decision_s", 0.0))
            if hasattr(agent, "_last_parallel_decision_s"):
                agent._last_parallel_decision_s = 0.0
        else:
            if hasattr(agent, "_last_parallel_decision_s"):
                agent._last_parallel_decision_s = 0.0

        for user in admit_users:
            if model_selection_dic[user].get("defer"):
                continue
            model_name = model_selection_dic[user]["model"]
            cell_id = cell_dic[user]
            md = agent.md_params[user]
            local_d = (agent.head_flops[model_name] * 1e-9
                       / (md["freq"] * md["cores"] * md["flops_per_cycle"]))
            local_e = md["power_coeff"] * md["freq"] ** 3 * local_d
            edge_unit = (agent.tail_flops[model_name] * 1e-9
                         / (es["cores"] * es["flops_per_cycle"]))
            agent.pipeline.admit(
                user, model_name, cell_id,
                agent._payload_bits(model_name),
                local_d, local_e, edge_unit, float(es["power_coeff"]))

        active_users = [
            u for u in agent.users if agent.pipeline.active[u] is not None
        ]
        radio_users = [
            u for u in active_users if agent.pipeline.needs_radio(u)
        ]
        for user in active_users:
            at = agent.pipeline.active[user]
            model_selection_dic[user] = {
                "model": at.model_name,
                "cell_rank": model_selection_dic.get(user, {}).get("cell_rank", 0),
            }
            cell_dic[user] = int(at.cell_id)
            task_dic[user]["delay_constraint"] = at.delay_constraint
            task_dic[user]["energy_constraint"] = at.energy_constraint
            task_dic[user]["delay_weight"] = at.delay_weight
            task_dic[user]["energy_weight"] = at.energy_weight

        total_overhead_dic = {
            u: {"delay": 0.0, "energy": 0.0} for u in agent.users
        }
        acc_realized_dic = {u: 0.0 for u in agent.users}
        reward_dic = {u: 0.0 for u in agent.users}
        phy_choice_dic = {u: agent.available_mcs[0] for u in agent.users}
        bandwidth_allocation_dic = {u: 0.0 for u in agent.users}
        gpu_allocation_dic = {u: 0.0 for u in agent.users}
        snr_true_dic = {u: 1e-12 for u in agent.users}
        snr_est_dic = {u: 1e-12 for u in agent.users}

        # Radio BCD: BW/MCS only for MDs with residual uplink bits (M^u).
        # GPU is FIFO: full pool to each cell's edge HOL, outside the radio BCD.
        snr_est_dic = {
            u: 10 ** (sinr_est_db_all_dic[u][cand_cells_dic[u][0]] / 10)
            for u in agent.users
        }
        snr_true_dic = {
            u: 10 ** (sinr_true_db_all_dic[u][cand_cells_dic[u][0]] / 10)
            for u in agent.users
        }
        custom_alloc = getattr(agent, "allocate_slot_resources", None)
        if callable(custom_alloc):
            # Same accounting bucket as BCD (CSV ``bcd_ms``): per-slot resource
            # allocation wall, whether EG slice enforcement or full BCD.
            t_alloc = time.perf_counter()
            if active_users:
                cell_for_snr = {
                    u: int(cell_dic[u]) for u in active_users if u in cell_dic}
                snr_est_dic.update(agent._apply_cell_association(
                    cell_for_snr, sinr_est_db_all_dic))
                snr_true_dic.update(agent._apply_cell_association(
                    cell_for_snr, sinr_true_db_all_dic))
            bw_c, gpu_c, phy_c = custom_alloc(
                radio_users, active_users, cell_dic,
                sinr_est_db_all_dic, model_selection_dic)
            bandwidth_allocation_dic.update(bw_c)
            gpu_allocation_dic = gpu_c
            phy_choice_dic.update(phy_c)
            if hasattr(agent, "bcd_time"):
                agent.bcd_time += time.perf_counter() - t_alloc
        elif radio_users:
            cell_dic_a = {}
            for u in radio_users:
                at = agent.pipeline.active[u]
                cell_dic_a[u] = int(at.cell_id)
                model_selection_dic[u] = {
                    "model": at.model_name,
                    "cell_rank": model_selection_dic.get(u, {}).get(
                        "cell_rank", 0),
                }
            model_a = {u: model_selection_dic[u] for u in radio_users}
            task_a = {u: task_dic[u] for u in radio_users}
            snr_est_dic.update(agent._apply_cell_association(
                cell_dic_a, sinr_est_db_all_dic))
            snr_true_dic.update(agent._apply_cell_association(
                cell_dic_a, sinr_true_db_all_dic))
            local_overhead_dic = agent.get_local_overhead(model_a)
            saved_users = agent.users
            agent.users = radio_users
            try:
                bcd_out = run_bcd_slot(
                    agent, task_a, model_a, trans_rate_dic,
                    local_overhead_dic, snr_est_dic, cell_dic_a)
            finally:
                agent.users = saved_users
            bandwidth_allocation_dic.update(bcd_out["bandwidth"])
            phy_choice_dic.update(bcd_out["phy_choice"])
            gpu_allocation_dic = agent.gpu_resource_allocation(
                task_dic, model_selection_dic, users=agent.users)
        else:
            gpu_allocation_dic = agent.gpu_resource_allocation(
                task_dic, model_selection_dic, users=agent.users)

        # Do not log mid-task predicted overhead into evaluation metrics;
        # BCD already used its own proxy inside run_bcd_slot. Realized e2e
        # delay/energy/reward are written only on task completion below.

        for user in agent.users:
            agent.instant_metrics[user]["mcs"].append(phy_choice_dic[user])
            agent.instant_metrics[user]["cell"].append(
                cell_dic.get(user, cand_cells_dic[user][0]))

        # Realized SINR uses this slot's grants and only the MDs that are on
        # the air. Decisions and the MCS above used the previous slot.
        radio_for_phy = []
        user_index = {user: i for i, user in enumerate(agent.users)}
        if hasattr(agent.sinr_trace, "realized_radio"):
            for user in radio_users:
                bw = float(bandwidth_allocation_dic.get(user, 0.0))
                if bw <= 1.0 or user not in cell_dic:
                    continue
                radio_for_phy.append((user_index[user], int(cell_dic[user]), bw))
            power_w = {}
            if radio_for_phy:
                sinr_db, power_w = agent.sinr_trace.realized_radio(
                    t, radio_for_phy)
                offset = float(getattr(agent, "sinr_offset_db", 0.0))
                for user in radio_users:
                    ui = user_index[user]
                    if ui not in sinr_db:
                        continue
                    snr_true_dic[user] = 10 ** ((sinr_db[ui] + offset) / 10.0)
            agent.sinr_trace.uplink_state = radio_for_phy

        for user in agent.users:
            if user in active_users and user in model_selection_dic:
                snr_db_u = 10 * np.log10(max(snr_true_dic[user], 1e-12))
                agent.instant_metrics[user]["bler"].append(agent.mcs_table.bler(
                    model_selection_dic[user]["model"],
                    phy_choice_dic[user], snr_db_u))
            else:
                agent.instant_metrics[user]["bler"].append(0.0)

        goodput_se = {}
        tx_power = {}
        for user in active_users:
            at = agent.pipeline.active[user]
            goodput_se[user] = agent._goodput_se(
                user, at.model_name, phy_choice_dic[user], snr_true_dic)
            ui = user_index.get(user)
            if ui is not None and hasattr(agent.sinr_trace, "realized_radio"):
                # Power follows the grant. A user who is not on the air this
                # slot contributes no transmit energy.
                tx_power[user] = 0.0
                for ue_local, _cell, _bw in radio_for_phy:
                    if ue_local == ui:
                        tx_power[user] = float(power_w.get(ui, 0.0))
                        break
            else:
                tx_power[user] = agent.md_params[user]["trans_power"]
            # Last uplink MCS κ^f: refresh while still radio-active.
            if agent.pipeline.needs_radio(user):
                at.last_mcs = int(phy_choice_dic[user])
        agent.pipeline.advance(
            agent.slot_duration,
            bandwidth_allocation_dic,
            gpu_allocation_dic,
            goodput_se,
            tx_power,
            agent.energy_budget,
            default_gpu_hz=float(es["freq"]),
        )
        done_tasks = agent.pipeline.pop_slot_completed()
        done_by_user = {}
        for task in done_tasks:
            done_by_user.setdefault(task.user, []).append(task)

        learn_snr, learn_mcs, learn_acc = {}, {}, {}
        learn_ctx, learn_msel, learn_rew = {}, {}, {}
        for user in agent.users:
            if user in done_by_user:
                task = done_by_user[user][-1]
                delay = task.e2e_delay
                energy = task.total_energy
                total_overhead_dic[user] = {"delay": delay, "energy": energy}
                snr_lin = snr_true_dic.get(user, 1e-12)
                snr_db = 10 * np.log10(max(snr_lin, 1e-12))
                mcs = task.last_mcs if task.last_mcs is not None else phy_choice_dic[user]
                acc = float(agent.mcs_table.accuracy(task.model_name, mcs, snr_db))
                if getattr(agent, "acc_noise_std", 0.0) > 0:
                    from omnis.exog import acc_noise
                    acc = float(np.clip(
                        acc + acc_noise(
                            getattr(agent, "seed", 0),
                            user,
                            getattr(task, "t_arrive", 0.0),
                            task.model_name,
                            agent.acc_noise_std),
                        0.0, 1.0))
                acc_realized_dic[user] = acc
                td = {
                    "delay_constraint": task.delay_constraint,
                    "energy_constraint": task.energy_constraint,
                    "delay_weight": task.delay_weight,
                    "energy_weight": task.energy_weight,
                }
                reward_dic[user] = agent.get_reward(
                    {user: td}, {user: acc},
                    {user: total_overhead_dic[user]})[user]
                learn_snr[user] = snr_lin
                learn_mcs[user] = mcs
                learn_acc[user] = acc
                learn_ctx[user] = context_dic.get(user, {
                    "delay_constraint": task.delay_constraint,
                    "energy_constraint": task.energy_constraint,
                    "transmission_rate": 0.0,
                    "energy_weight": task.energy_weight,
                    "delay_weight": task.delay_weight,
                })
                learn_msel[user] = {
                    "model": task.model_name,
                    "cell_rank": model_selection_dic.get(user, {}).get(
                        "cell_rank", 0),
                }
                learn_rew[user] = reward_dic[user]
            # Mid-task slots: leave reward/acc/overhead at 0 (no learning label).

            agent.backlog[user] = float(agent.pipeline.composite_backlog(user))

        queue_info_dic = {}
        for user in agent.users:
            served = 1.0 if user in done_by_user else 0.0
            queue_info_dic[user] = {
                "backlog": agent.backlog[user],
                "energy_queue": agent.energy_queue[user],
                "arrivals": float(task_dic[user]["n_arrivals"]),
                "served": served,
            }
        agent.get_instant_metrics(
            task_dic, total_overhead_dic, reward_dic,
            acc_realized_dic, queue_info_dic)

        if not hasattr(agent, "_last_bandwidth"):
            agent._last_bandwidth = {}
        if not hasattr(agent, "_last_gpu"):
            agent._last_gpu = {}
        for u, bw in bandwidth_allocation_dic.items():
            if bw > 0:
                agent._last_bandwidth[u] = bw
        for u, g in gpu_allocation_dic.items():
            if g > 0:
                agent._last_gpu[u] = g
        agent._cell_compute_state = update_cell_compute_state(
            {u: cell_dic[u] for u in active_users} if active_users else {},
            {u: gpu_allocation_dic[u] for u in active_users},
            agent.num_cells,
            agent.es_params["freq"],
            q_edge_by_cell={
                c: agent.pipeline.compute_queue_len(c)
                for c in range(agent.num_cells)
            },
        )

        greedy_learn = _is_greedy_mab(agent)
        can_learn = learn_acc and (
            getattr(agent, "algo", None) == "causal"
            or hasattr(agent, "optimizers")
            or hasattr(agent, "update_gp")
            or greedy_learn
        )
        if can_learn:
            learn_payload = {
                "algo": getattr(agent, "algo", None),
                "snr": learn_snr,
                "mcs": learn_mcs,
                "acc": learn_acc,
                "context": learn_ctx,
                "model_selection": learn_msel,
                "reward": learn_rew,
                "users": list(learn_acc.keys()),
            }
            if int(getattr(agent, "learn_delay_slots", 0) or 0) > 0:
                agent._pending_learn = learn_payload
            elif greedy_learn:
                agent.learn_after_slot(learn_payload)
                if hasattr(agent, "update_time"):
                    agent.update_time += float(
                        getattr(agent, "_last_parallel_update_s", 0.0))
            else:
                apply_learn_payload(agent, learn_payload)
                if hasattr(agent, "update_time"):
                    agent.update_time += float(
                        getattr(agent, "_last_parallel_update_s", 0.0))

    if (int(getattr(agent, "learn_delay_slots", 0) or 0) > 0
            and getattr(agent, "_pending_learn", None) is not None):
        flush_pending_learn(agent)
        if hasattr(agent, "update_time"):
            agent.update_time += float(
                getattr(agent, "_last_parallel_update_s", 0.0))
    agent.get_average_and_std_metrics()
