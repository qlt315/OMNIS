import time

import numpy as np
from scipy.special import erf

from omnis.causal_gp import ResidualGP


class CausalMAB:
    """Contextual bandit over (model, cell) with a residual Acc GP from observations."""

    def __init__(self, scm, length_scales, signal_var, noise_var, beta,
                 penalty_gain=1.5, acquisition='ucb', use_prior=False, shared=True,
                 lyapunov_v=1.0, drift_gain=1.0, w_acc=1.0, init_random=20,
                 empty_prior_std=1.0, explore_slots=20, max_obs=600,
                 acc_upgrade_snr_db=4.0, acc_upgrade_backlog_tanh=0.45,
                 acc_upgrade_bonus=0.10, feas_margin=1.0, centralized=False):
        self.scm = scm
        self.beta = beta
        self.penalty_gain = penalty_gain
        self.acquisition = acquisition
        # Acc-table prior is banned; flag kept for compatibility (always off).
        self.use_prior = bool(use_prior) and bool(scm._acc_prior)
        self.shared = shared
        self.lyapunov_v = lyapunov_v
        self.drift_gain = drift_gain
        self.w_acc = w_acc
        # Random (model, cell) until this many pooled GP observations (乱搞),
        # or until explore_slots selection rounds — whichever keeps exploring longer.
        self.init_random = int(init_random)
        self.explore_slots = int(explore_slots)
        self.min_obs_per_model = 12
        self._select_slots = 0
        self._model_obs = [0 for _ in scm.models]
        # Lyapunov-feasible accuracy upgrade: bias toward wider models when
        # SINR is good and the bit queue is light (does not change get_reward).
        self.acc_upgrade_snr_db = float(acc_upgrade_snr_db)
        self.acc_upgrade_backlog_tanh = float(acc_upgrade_backlog_tanh)
        self.acc_upgrade_bonus = float(acc_upgrade_bonus)
        # Require predicted sojourn/energy ≤ margin * constraint (margin≤1),
        # with a small inflation on sojourn to cover prediction optimism.
        self.feas_margin = float(np.clip(feas_margin, 0.5, 1.0))
        # Centralized controller (CTO): charge the sum of per-MD work plus the
        # full GP predict. Distributed MDs charge max(local) plus GP wall / U.
        self.centralized = bool(centralized)
        # Small cushion for SINR error. The admit-slot grant wait and the
        # broadcast association share are already inside the sojourn forecast.
        # Slightly inside the raw delay limit. GDO admits whenever the
        # predicted sojourn meets the limit, so the same heavy branch is
        # eligible here unless the forecast is already against the constraint.
        self.sojourn_guard = 1.0
        self.sojourn_guard_hi_snr = 1.0
        self.sojourn_guard_snr_db = 6.5
        self.heavy_delay_factor = 1.0
        self.light_delay_factor = 1.0
        self._channel_tier = {
            3: 0.0, 6: 0.55, 12: 1.0,
        }
        max_obs = int(max_obs)
        self._gp_args = dict(
            prior_mean_fn=self._prior_at,
            length_scales=length_scales,
            signal_var=signal_var,
            noise_var=noise_var,
            empty_prior_std=empty_prior_std,
            max_obs=max_obs,
        )
        self.gp = ResidualGP(**self._gp_args)
        self._gps = {}
        self._pending = {}  # per-MD (snr_db, model_idx, cell_id) of the last arm
        self.last_parallel_decision_s = 0.0

    def _gp_for(self, user):
        if self.shared:
            return self.gp
        if user not in self._gps:
            self._gps[user] = ResidualGP(**self._gp_args)
        return self._gps[user]

    def _make_x(self, snr_db, arm_idx, mcs_idx):
        """Features are SINR, quantization, width, and coding rate.

        MCS indices are not ordered by code rate (a higher index can be a
        lower rate), so the GP coordinate is the rate itself. Accuracy is
        Acc(branch, rate, SINR); nearby indices are not nearby rates.
        """
        quant_flag, channels = self.scm.arm_feature(arm_idx)
        rate = float(self.scm.mcs_table.code_rate[int(mcs_idx)])
        return np.array([snr_db, quant_flag, channels, rate])

    def _prior_at(self, x):
        """Uninformative Acc prior (0). Never reads mcs_table.accuracy."""
        if not self.use_prior:
            return 0.0
        snr_db, quant_flag, channels, mcs_idx = x
        model_name = self.scm.feature_to_name(quant_flag, channels)
        return self.scm.acc_prior_mean(snr_db, model_name, int(round(mcs_idx)))

    def predict_mcs(self, snr_db, model_name, task, predict_overheads,
                    model_idx=None, overhead_parts=None, cell_id=None):
        """ILLA MCS forward sim: BLER/SE (+ QoS); no Acc-GP / Acc-table inside.

        When ``overhead_parts`` is provided (including cell-specific GPU), MCS
        search is vectorized — same cost class as UCB ``forward_sim_mcs``.
        """
        del model_idx  # Acc-GP must not run inside the MCS loop
        if overhead_parts is not None:
            return self._predict_mcs_vectorized(
                snr_db, model_name, task, overhead_parts)
        bler_t = self.scm.bler_target()
        feas = []
        best_infeas, best_infeas_score = None, -np.inf
        for mcs in self.scm.available_mcs:
            service_hat, _, energy_hat = predict_overheads(
                model_name, mcs, snr_db=snr_db, cell_id=cell_id)
            if (service_hat <= task['delay_constraint']
                    and energy_hat <= task['energy_constraint']):
                bler = self.scm.bler(model_name, mcs, snr_db)
                feas.append((mcs, bler, self.scm.se(mcs)))
            else:
                score = (
                    task['delay_weight'] * erf(task['delay_constraint'] - service_hat)
                    + task['energy_weight'] * erf(task['energy_constraint'] - energy_hat))
                if score > best_infeas_score:
                    best_infeas, best_infeas_score = mcs, score
        if feas:
            under = [t for t in feas if t[1] <= bler_t]
            pool = under if under else feas
            return max(pool, key=lambda t: t[2])[0]
        return best_infeas

    def _predict_mcs_vectorized(self, snr_db, model_name, task, overhead_parts):
        """Vectorized ILLA over all MCS for one (model, snr, user-base)."""
        local_d, local_e, edge_d, edge_e, payload, bw, backlog, p_tx = overhead_parts
        bler, goodput = self.scm.mcs_table.bler_goodput_all_mcs(model_name, snr_db)
        mcs_idx = np.asarray(self.scm.available_mcs, dtype=int)
        se = self.scm.mcs_table._se_arr
        rates = bw * np.maximum(goodput, 1e-12)
        trans_d = payload / np.maximum(rates, 1e-12)
        slot = float(task.get("slot_duration", 1.0))
        grant = 0.0
        if local_d > 1e-12 and slot > 0.0:
            frac = float(local_d) % slot
            if frac > 1e-9:
                grant = slot - frac
        service = local_d + trans_d + edge_d + grant
        energy = local_e + p_tx * trans_d + edge_e
        d_c = task['delay_constraint']
        e_c = task['energy_constraint']
        feas = (service <= d_c) & (energy <= e_c)
        bler_t = self.scm.bler_target()
        if np.any(feas):
            under = feas & (bler <= bler_t)
            mask = under if np.any(under) else feas
            best_j = int(np.argmax(np.where(mask, se, -np.inf)))
            return int(mcs_idx[best_j])
        score = (task['delay_weight'] * erf(d_c - service)
                 + task['energy_weight'] * erf(e_c - energy))
        return int(mcs_idx[int(np.argmax(score))])

    def select_arm(self, user, top_cells, sinr_db_by_cell, task, predict_overheads):
        """Score every joint (model, cell_rank) arm and return the best."""
        return self.select_arms_batch(
            [(user, top_cells, sinr_db_by_cell, task, predict_overheads)])[user]

    def _n_obs(self):
        if self.shared:
            return len(self.gp)
        return sum(len(g) for g in self._gps.values())

    def _score_user_arms(self, user, top_cells, sinr_db_by_cell, task,
                         predict_overheads, drift_score, overhead_parts_by_model=None):
        """Per-MD analytic work: M×L vectorized ILLA + overhead hats (no GP).

        ``overhead_parts_by_model`` may be:
          * ``None`` — scalar MCS loop (legacy),
          * ``dict[model_name -> parts]`` — shared GPU (no cell),
          * ``callable(model_name, cell_id) -> parts`` — cell-aware vectorized ILLA.
        """
        L = len(top_cells)
        mcs_hats = []
        overhead_hats = []
        arm_meta = []
        xs = []
        parts_fn = overhead_parts_by_model if callable(overhead_parts_by_model) else None
        parts_map = None if parts_fn else overhead_parts_by_model
        for model_idx, model in enumerate(self.scm.models):
            name = model['name']
            for cell_rank in range(L):
                cell_id = top_cells[cell_rank]
                snr_db = sinr_db_by_cell[cell_id]
                if parts_fn is not None:
                    parts = parts_fn(name, cell_id)
                elif parts_map is not None:
                    parts = parts_map.get(name)
                else:
                    parts = None
                mcs_hat = self.predict_mcs(
                    snr_db, name, task, predict_overheads,
                    model_idx=model_idx, overhead_parts=parts,
                    cell_id=None if parts is not None else cell_id)
                mcs_hats.append(mcs_hat)
                overhead_hats.append(
                    predict_overheads(name, mcs_hat, snr_db=snr_db,
                                      cell_id=cell_id))
                xs.append(self._make_x(snr_db, model_idx, mcs_hat))
                arm_meta.append((model_idx, cell_id, snr_db))
        return user, task, mcs_hats, overhead_hats, drift_score, arm_meta, xs

    def _acc_upgrade_uplift(self, task, snr_db, model_idx):
        """Extra utility [same units as get_reward] for wider models when safe."""
        if self.acc_upgrade_bonus <= 0.0:
            return 0.0
        backlog = float(task.get('backlog_bits', 0.0))
        scale = float(task.get('dpp_bit_scale', 1.0))
        if scale <= 0.0:
            return 0.0
        if snr_db < self.acc_upgrade_snr_db:
            return 0.0
        if float(np.tanh(backlog / scale)) >= self.acc_upgrade_backlog_tanh:
            return 0.0
        _quant, channels = self.scm.arm_feature(model_idx)
        tier = self._channel_tier.get(int(channels), 0.0) + 0.20 * float(_quant)
        # Prefer Standard12 / 12-ch partitions when the upgrade conditions hold.
        name = self.scm.models[model_idx]["name"]
        if name == "Standard12":
            tier += 1.5
        elif name.endswith("12"):
            tier += 0.8
        if tier <= 0.0:
            return 0.0
        return self.acc_upgrade_bonus * tier

    def _delay_budget_factor(self, model_idx):
        name = self.scm.models[model_idx]["name"]
        if name in ("Box12", "Standard6", "Standard12"):
            return float(self.heavy_delay_factor)
        return float(self.light_delay_factor)

    def _local_argmax(self, user, task, mcs_hats, overhead_hats, drift_score,
                      arm_meta, acc_scores):
        """Among QoS-feasible arms, maximize V·u + drift.

        The erf in u is the soft penalty. The gate is the task constraint:
        predicted sojourn and energy must sit inside this task's limits.
        If every arm misses, take the smallest violation.
        """
        d_lim = float(task['delay_constraint']) * self.feas_margin
        e_lim = float(task['energy_constraint']) * self.feas_margin
        best_feas = (-np.inf, None)
        best_any = ((-np.inf, -np.inf), None)
        for arm_idx in range(len(arm_meta)):
            model_idx, cell_id, snr_db = arm_meta[arm_idx]
            _, sojourn_hat, energy_hat = overhead_hats[arm_idx]
            reward_hat = (self.w_acc * acc_scores[arm_idx]
                          + self.penalty_gain * task['delay_weight']
                          * erf(float(task['delay_constraint']) - sojourn_hat)
                          + self.penalty_gain * task['energy_weight']
                          * erf(float(task['energy_constraint']) - energy_hat)
                          + self._acc_upgrade_uplift(task, snr_db, model_idx))
            val = self.lyapunov_v * reward_hat
            if drift_score is not None:
                val += self.drift_gain * drift_score(
                    self.scm.models[model_idx]['name'],
                    mcs_hats[arm_idx], energy_hat, snr_db=snr_db,
                    cell_id=cell_id)
            slack = (min(0.0, d_lim - sojourn_hat)
                     + min(0.0, e_lim - energy_hat))
            if (slack, val) > best_any[0]:
                best_any = ((slack, val), arm_idx)
            if sojourn_hat <= d_lim and energy_hat <= e_lim and val > best_feas[0]:
                best_feas = (val, arm_idx)
        best_arm = best_feas[1] if best_feas[1] is not None else best_any[1]
        model_idx, cell_id, snr_db = arm_meta[best_arm]
        self._pending[user] = (snr_db, model_idx, cell_id)
        return model_idx, cell_id

    def select_arms_batch(self, requests, advance_slot=True):
        """Batched joint-arm selection with parallel (max-agent) timing.

        requests: iterable of
            (user, top_cells, sinr_db_by_cell, task, predict_overheads
             [, drift_score [, overhead_parts_by_model]]).
        Early observations (``_n_obs < init_random``): uniform random arms.

        Sets ``last_parallel_decision_s`` to the distributed parallel cost
        (max over MDs, GP wall / U) unless ``centralized`` is set, in which
        case the controller is charged the sequential sum plus the full GP.
        """
        requests = list(requests)
        num_models = len(self.scm.models)
        explore = (
            self._select_slots < self.explore_slots
            and min(self._model_obs) < self.min_obs_per_model
        )
        if explore:
            selected = {}
            user_times = []
            for request in requests:
                t_u = time.time()
                user = request[0]
                top_cells = request[1]
                sinr_db_by_cell = request[2]
                task = request[3]
                predict_overheads = request[4]
                drift_score = request[5] if len(request) > 5 else None
                parts = request[6] if len(request) > 6 else None
                _u, _t, mcs_hats, overhead_hats, _d, arm_meta, _xs = self._score_user_arms(
                    user, top_cells, sinr_db_by_cell, task,
                    predict_overheads, drift_score, parts)
                d_lim = float(task['delay_constraint']) * self.feas_margin
                e_lim = float(task['energy_constraint']) * self.feas_margin
                guard0 = float(getattr(self, "sojourn_guard", 1.0))
                guard_hi = float(getattr(self, "sojourn_guard_hi_snr", guard0))
                snr_hi = float(getattr(self, "sojourn_guard_snr_db", 6.0))
                feas_idx = []
                for i, oh in enumerate(overhead_hats):
                    mid, _cid, snr_db = arm_meta[i]
                    is_std12 = self.scm.models[mid]["name"] == "Standard12"
                    g = guard_hi if (is_std12 and snr_db >= snr_hi) else guard0
                    d_arm = d_lim * self._delay_budget_factor(mid)
                    if oh[1] * g <= d_arm and oh[2] <= e_lim:
                        feas_idx.append(i)
                if not feas_idx:
                    pick = int(np.argmin([oh[1] for oh in overhead_hats]))
                else:
                    # Fewest observations first, and the best predicted cell
                    # of that branch, so every feasible branch is identified.
                    best_cell = {}
                    for i in feas_idx:
                        mid = arm_meta[i][0]
                        soj = overhead_hats[i][1]
                        prev = best_cell.get(mid)
                        if prev is None or soj < prev[0]:
                            best_cell[mid] = (soj, i)
                    mid = min(best_cell, key=lambda m: (self._model_obs[m], m))
                    pick = best_cell[mid][1]
                model_idx, cell_id, snr_db = arm_meta[pick]
                selected[user] = (model_idx, cell_id)
                self._pending[user] = (snr_db, model_idx, cell_id)
                user_times.append(time.time() - t_u)
            if advance_slot:
                self._select_slots += 1
            self._finish_decision_time(user_times, 0.0)
            return selected
        if advance_slot:
            self._select_slots += 1

        packed = []
        user_times = []
        for request in requests:
            t_u = time.time()
            user = request[0]
            top_cells = request[1]
            sinr_db_by_cell = request[2]
            task = request[3]
            predict_overheads = request[4]
            drift_score = request[5] if len(request) > 5 else None
            parts = request[6] if len(request) > 6 else None
            packed.append(self._score_user_arms(
                user, top_cells, sinr_db_by_cell, task,
                predict_overheads, drift_score, parts))
            user_times.append(time.time() - t_u)

        arms_per_user = len(packed[0][5]) if packed else 0
        flat_xs = []
        for _user, _task, _mcs, _oh, _drift, _meta, xs in packed:
            flat_xs.extend(xs)

        t_gp = time.time()
        if self.shared:
            mu, std = self.gp.predict(np.array(flat_xs))
            acc_score = self._acquire(mu, std)
            scores = [
                acc_score[u_idx * arms_per_user:(u_idx + 1) * arms_per_user]
                for u_idx in range(len(packed))]
        else:
            scores = []
            for u_idx, (user, _, _, _, _, _, _) in enumerate(packed):
                xs_u = np.array(
                    flat_xs[u_idx * arms_per_user:(u_idx + 1) * arms_per_user])
                mu, std = self._gp_for(user).predict(xs_u)
                scores.append(self._acquire(mu, std))
        gp_wall = time.time() - t_gp

        selected = {}
        for u_idx, (user, task, mcs_hats, overhead_hats, drift_score,
                    arm_meta, _xs) in enumerate(packed):
            t_u = time.time()
            selected[user] = self._local_argmax(
                user, task, mcs_hats, overhead_hats, drift_score,
                arm_meta, scores[u_idx])
            user_times[u_idx] += time.time() - t_u

        self._finish_decision_time(user_times, gp_wall)
        return selected

    def _finish_decision_time(self, user_times, gp_wall):
        """Distributed: max MD time + GP wall/U. Centralized: sum + full GP."""
        if not user_times:
            self.last_parallel_decision_s = 0.0
            return
        if self.centralized:
            self.last_parallel_decision_s = float(sum(user_times) + gp_wall)
            return
        n = max(len(user_times), 1)
        self.last_parallel_decision_s = max(user_times) + float(gp_wall) / n

    def _acquire(self, mu, std):
        if self.acquisition == 'ts':
            return mu + std * np.random.randn(len(mu))
        return mu + self.beta * std

    def register_outcome(self, user, mcs_realized, acc_obs, do_update=True):
        """Register the realized interventional observation in the GP.

        If ``do_update`` is False (``mab_no_update`` / freeze), drop the pending
        arm without calling ``add`` so selection keeps the frozen posterior.
        """
        snr_db, model_idx, _cell_id = self._pending.pop(user)
        self._model_obs[model_idx] += 1
        if do_update:
            self._gp_for(user).add(self._make_x(snr_db, model_idx, mcs_realized), acc_obs)

    def register_outcomes_batch(self, records, do_update=True):
        """Register one slot of realized outcomes: [(user, mcs_idx, acc_obs)]."""
        for user, mcs_realized, acc_obs in records:
            self.register_outcome(user, mcs_realized, acc_obs, do_update=do_update)

    def prediction_errors(self, records):
        """Abs accuracy errors vs prior / current posterior at the pending arm.

        Call **before** ``register_outcomes_batch`` (uses ``_pending``).
        Returns (mean_|acc-prior|, mean_|acc-posterior|) over users in ``records``.
        With empty prior, prior error ≈ |acc| (uninformative baseline).
        """
        prior_errs = []
        post_errs = []
        if self.shared and records:
            xs = []
            accs = []
            priors = []
            for user, mcs_realized, acc_obs in records:
                snr_db, model_idx, _cell_id = self._pending[user]
                x = self._make_x(snr_db, model_idx, mcs_realized)
                xs.append(x)
                accs.append(float(acc_obs))
                priors.append(float(self._prior_at(x)))
            mu, _std = self.gp.predict(np.asarray(xs, dtype=float))
            prior_errs = [abs(a - p) for a, p in zip(accs, priors)]
            post_errs = [abs(a - float(m)) for a, m in zip(accs, mu)]
        else:
            for user, mcs_realized, acc_obs in records:
                snr_db, model_idx, _cell_id = self._pending[user]
                x = self._make_x(snr_db, model_idx, mcs_realized)
                prior = float(self._prior_at(x))
                mu, _std = self._gp_for(user).predict(
                    np.asarray(x, dtype=float)[None, :])
                post = float(np.asarray(mu).ravel()[0])
                prior_errs.append(abs(float(acc_obs) - prior))
                post_errs.append(abs(float(acc_obs) - post))
        return float(np.mean(prior_errs)), float(np.mean(post_errs))

    def __len__(self):
        if self.shared:
            return len(self.gp)
        return sum(len(g) for g in self._gps.values())
