import time

import numpy as np
from scipy.special import erf

from omnis.causal_gp import ResidualGP


class CausalMAB:
    """Shared causal contextual bandit for dynamic split-DNN branch selection.

    A single global residual GP learns the accuracy mechanism
    P(acc | do(Model, MCS), SINR) from **observations only** (no Acc-table
    prior). Early slots with empty / sparse GP explore randomly or via large
    UCB/TS uncertainty. Each MD composes the GP Acc estimate with analytic
    QoS + drift into a reward-aligned score matching ``get_reward``.

    MCS forward simulation uses ILLA on BLER/SE (+ QoS erf when infeasible);
    Acc-table tie-breaks are banned. GP Acc is scored once per joint arm after
    MCS is chosen — never inside the MCS loop.

    Joint arms are (model, cell_rank) **inside a coarse Top-L** pruned by
    MD-visible radio + broadcast compute (``omnis.assoc_info.coarse_rank_cells``).
    Pruning reduces exploration; joint learning is still required because Acc,
    link adaptation, edge GPU share, and queues are coupled in the Lyapunov
    objective — coarse scores are not optimality certificates.

    Parallel decision model (distributed MDs): ``last_parallel_decision_s`` is
    max over per-MD analytic work (+ shared GP predict amortized by /U).
    """

    def __init__(self, scm, length_scales, signal_var, noise_var, beta,
                 penalty_gain=1.5, acquisition='ucb', use_prior=False, shared=True,
                 lyapunov_v=1.0, drift_gain=1.0, w_acc=1.0, init_random=20,
                 empty_prior_std=1.0, explore_slots=20, max_obs=600,
                 acc_upgrade_snr_db=4.0, acc_upgrade_backlog_tanh=0.45,
                 acc_upgrade_bonus=0.10):
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
        self._select_slots = 0
        # Lyapunov-feasible accuracy upgrade: bias toward wider models when
        # SINR is good and the bit queue is light (does not change get_reward).
        self.acc_upgrade_snr_db = float(acc_upgrade_snr_db)
        self.acc_upgrade_backlog_tanh = float(acc_upgrade_backlog_tanh)
        self.acc_upgrade_bonus = float(acc_upgrade_bonus)
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
        quant_flag, channels = self.scm.arm_feature(arm_idx)
        return np.array([snr_db, quant_flag, channels, float(mcs_idx)])

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
        trans_d = payload / rates
        service = local_d + trans_d + edge_d
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
        if tier <= 0.0:
            return 0.0
        return self.acc_upgrade_bonus * tier

    def _local_argmax(self, user, task, mcs_hats, overhead_hats, drift_score,
                      arm_meta, acc_scores):
        """Per-MD local composition + argmax given Acc scores for its arms."""
        best_val, best_arm = -np.inf, 0
        for arm_idx in range(len(arm_meta)):
            model_idx, cell_id, snr_db = arm_meta[arm_idx]
            _, sojourn_hat, energy_hat = overhead_hats[arm_idx]
            reward_hat = (self.w_acc * acc_scores[arm_idx]
                          + self.penalty_gain * task['delay_weight']
                          * erf(task['delay_constraint'] - sojourn_hat)
                          + self.penalty_gain * task['energy_weight']
                          * erf(task['energy_constraint'] - energy_hat))
            reward_hat += self._acc_upgrade_uplift(task, snr_db, model_idx)
            val = self.lyapunov_v * reward_hat
            if drift_score is not None:
                val += self.drift_gain * drift_score(
                    self.scm.models[model_idx]['name'],
                    mcs_hats[arm_idx], energy_hat, snr_db=snr_db)
            if val > best_val:
                best_val, best_arm = val, arm_idx
        model_idx, cell_id, snr_db = arm_meta[best_arm]
        self._pending[user] = (snr_db, model_idx, cell_id)
        return model_idx, cell_id

    def select_arms_batch(self, requests):
        """Batched joint-arm selection with parallel (max-agent) timing.

        requests: iterable of
            (user, top_cells, sinr_db_by_cell, task, predict_overheads
             [, drift_score [, overhead_parts_by_model]]).
        Early observations (``_n_obs < init_random``): uniform random arms.

        Sets ``last_parallel_decision_s`` = max over MD analytic times, plus
        shared GP predict wall / U (amortized).
        """
        requests = list(requests)
        num_models = len(self.scm.models)
        explore = (self._select_slots < self.explore_slots
                   or self._n_obs() < self.init_random)
        if explore:
            selected = {}
            user_times = []
            for request in requests:
                t_u = time.time()
                user, top_cells, sinr_db_by_cell = request[0], request[1], request[2]
                L = len(top_cells)
                model_idx = int(np.random.randint(0, num_models))
                cell_rank = int(np.random.randint(0, L))
                cell_id = top_cells[cell_rank]
                snr_db = sinr_db_by_cell[cell_id]
                selected[user] = (model_idx, cell_id)
                self._pending[user] = (snr_db, model_idx, cell_id)
                user_times.append(time.time() - t_u)
            self._select_slots += 1
            self.last_parallel_decision_s = max(user_times) if user_times else 0.0
            return selected
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
            gp_s = (time.time() - t_gp) / max(len(packed), 1)
        else:
            scores = []
            for u_idx, (user, _, _, _, _, _, _) in enumerate(packed):
                xs_u = np.array(
                    flat_xs[u_idx * arms_per_user:(u_idx + 1) * arms_per_user])
                mu, std = self._gp_for(user).predict(xs_u)
                scores.append(self._acquire(mu, std))
            gp_s = time.time() - t_gp  # per-MD GPs: count full wall once as max proxy

        selected = {}
        for u_idx, (user, task, mcs_hats, overhead_hats, drift_score,
                    arm_meta, _xs) in enumerate(packed):
            t_u = time.time()
            selected[user] = self._local_argmax(
                user, task, mcs_hats, overhead_hats, drift_score,
                arm_meta, scores[u_idx])
            user_times[u_idx] += time.time() - t_u

        local_max = max(user_times) if user_times else 0.0
        self.last_parallel_decision_s = local_max + (gp_s if self.shared else 0.0)
        if not self.shared:
            # Non-shared: GP predict already in per-user times if done inside loop;
            # here GP was sequential — use max(local, gp/U) style: add gp/U.
            self.last_parallel_decision_s = local_max + gp_s / max(len(packed), 1)
        return selected

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
