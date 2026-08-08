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
    Acc-table tie-breaks are banned. After the GP has data, optional Acc
    estimates come from the GP posterior — never from the table.

    Joint arms are (model, cell_rank); cell_rank indexes the UE's Top-L
    strongest cells at the current slot.
    """

    def __init__(self, scm, length_scales, signal_var, noise_var, beta,
                 penalty_gain=1.5, acquisition='ucb', use_prior=False, shared=True,
                 lyapunov_v=1.0, drift_gain=1.0, w_acc=1.0, init_random=20,
                 empty_prior_std=1.0, explore_slots=20, max_obs=600):
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

    def _gp_acc_mean(self, snr_db, model_idx, mcs_idx, user=None):
        """Posterior Acc mean if the GP has data; else None (no table fallback)."""
        gp = self._gp_for(user) if user is not None else self.gp
        if len(gp) == 0:
            return None
        x = self._make_x(snr_db, model_idx, mcs_idx)
        mu, _std = gp.predict(np.asarray(x, dtype=float)[None, :])
        return float(np.asarray(mu).ravel()[0])

    def predict_mcs(self, snr_db, model_name, task, predict_overheads, model_idx=None):
        """ILLA MCS forward sim: BLER/SE (+ QoS); no Acc-table tie-break.

        Prefer QoS-feasible MCS with BLER <= bler_target (highest SE). If the
        residual GP has observations and ``model_idx`` is given, SE ties may
        break on GP Acc estimate — never on the Acc table. Infeasible MCS are
        scored by QoS erf only (optionally + GP Acc).
        """
        bler_t = self.scm.bler_target()
        feas = []
        best_infeas, best_infeas_score = None, -np.inf
        name_to_idx = {m['name']: i for i, m in enumerate(self.scm.models)}
        mid = model_idx if model_idx is not None else name_to_idx.get(model_name)
        for mcs in self.scm.available_mcs:
            service_hat, _, energy_hat = predict_overheads(model_name, mcs, snr_db=snr_db)
            gp_acc = (self._gp_acc_mean(snr_db, mid, mcs)
                      if mid is not None else None)
            acc_hat = 0.0 if gp_acc is None else gp_acc
            if (service_hat <= task['delay_constraint']
                    and energy_hat <= task['energy_constraint']):
                bler = self.scm.bler(model_name, mcs, snr_db)
                feas.append((mcs, bler, self.scm.se(mcs), acc_hat))
            else:
                score = (acc_hat
                         + task['delay_weight'] * erf(task['delay_constraint'] - service_hat)
                         + task['energy_weight'] * erf(task['energy_constraint'] - energy_hat))
                if score > best_infeas_score:
                    best_infeas, best_infeas_score = mcs, score
        if feas:
            under = [t for t in feas if t[1] <= bler_t]
            pool = under if under else feas
            # Primary: SE; secondary: GP Acc if available (not table).
            return max(pool, key=lambda t: (t[2], t[3]))[0]
        return best_infeas

    def select_arm(self, user, top_cells, sinr_db_by_cell, task, predict_overheads):
        """Score every joint (model, cell_rank) arm and return the best."""
        return self.select_arms_batch(
            [(user, top_cells, sinr_db_by_cell, task, predict_overheads)])[user]

    def _n_obs(self):
        if self.shared:
            return len(self.gp)
        return sum(len(g) for g in self._gps.values())

    def select_arms_batch(self, requests):
        """Batched joint-arm selection: one GP posterior solve per slot.

        requests: iterable of
            (user, top_cells, sinr_db_by_cell, task, predict_overheads[, drift_score]).
        Early observations (``_n_obs < init_random``): uniform random arms (乱搞).
        """
        num_models = len(self.scm.models)
        # Burn-in: exploratory / random policy before the Acc GP is trustworthy.
        explore = (self._select_slots < self.explore_slots
                   or self._n_obs() < self.init_random)
        if explore:
            selected = {}
            for request in requests:
                user, top_cells, sinr_db_by_cell = request[0], request[1], request[2]
                L = len(top_cells)
                model_idx = int(np.random.randint(0, num_models))
                cell_rank = int(np.random.randint(0, L))
                cell_id = top_cells[cell_rank]
                snr_db = sinr_db_by_cell[cell_id]
                selected[user] = (model_idx, cell_id)
                self._pending[user] = (snr_db, model_idx, cell_id)
            self._select_slots += 1
            return selected
        self._select_slots += 1

        flat_xs = []
        meta = []
        for request in requests:
            user, top_cells, sinr_db_by_cell, task, predict_overheads = request[:5]
            drift_score = request[5] if len(request) > 5 else None
            L = len(top_cells)
            mcs_hats = []
            overhead_hats = []
            arm_meta = []
            for model_idx, model in enumerate(self.scm.models):
                for cell_rank in range(L):
                    cell_id = top_cells[cell_rank]
                    snr_db = sinr_db_by_cell[cell_id]
                    mcs_hat = self.predict_mcs(
                        snr_db, model['name'], task, predict_overheads,
                        model_idx=model_idx)
                    mcs_hats.append(mcs_hat)
                    overhead_hats.append(predict_overheads(model['name'], mcs_hat, snr_db=snr_db))
                    flat_xs.append(self._make_x(snr_db, model_idx, mcs_hat))
                    arm_meta.append((model_idx, cell_id, snr_db))
            meta.append((user, task, mcs_hats, overhead_hats, drift_score, arm_meta))

        arms_per_user = len(meta[0][5]) if meta else 0

        if self.shared:
            mu, std = self.gp.predict(np.array(flat_xs))
            acc_score = self._acquire(mu, std)
            scores = [acc_score[u_idx * arms_per_user:(u_idx + 1) * arms_per_user]
                      for u_idx in range(len(meta))]
        else:
            scores = []
            for u_idx, (user, _, _, _, _, _) in enumerate(meta):
                xs_u = np.array(flat_xs[u_idx * arms_per_user:(u_idx + 1) * arms_per_user])
                mu, std = self._gp_for(user).predict(xs_u)
                scores.append(self._acquire(mu, std))

        selected = {}
        for u_idx, (user, task, mcs_hats, overhead_hats, drift_score, arm_meta) in enumerate(meta):
            best_val, best_arm = -np.inf, 0
            for arm_idx in range(arms_per_user):
                model_idx, cell_id, snr_db = arm_meta[arm_idx]
                _, sojourn_hat, energy_hat = overhead_hats[arm_idx]
                reward_hat = (self.w_acc * scores[u_idx][arm_idx]
                              + self.penalty_gain * task['delay_weight']
                              * erf(task['delay_constraint'] - sojourn_hat)
                              + self.penalty_gain * task['energy_weight']
                              * erf(task['energy_constraint'] - energy_hat))
                val = self.lyapunov_v * reward_hat
                if drift_score is not None:
                    val += self.drift_gain * drift_score(
                        self.scm.models[model_idx]['name'],
                        mcs_hats[arm_idx], energy_hat, snr_db=snr_db)
                if val > best_val:
                    best_val, best_arm = val, arm_idx
            model_idx, cell_id, snr_db = arm_meta[best_arm]
            selected[user] = (model_idx, cell_id)
            self._pending[user] = (snr_db, model_idx, cell_id)
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
        for user, mcs_realized, acc_obs in records:
            snr_db, model_idx, _cell_id = self._pending[user]
            x = self._make_x(snr_db, model_idx, mcs_realized)
            prior = float(self._prior_at(x))
            mu, _std = self._gp_for(user).predict(np.asarray(x, dtype=float)[None, :])
            post = float(np.asarray(mu).ravel()[0])
            prior_errs.append(abs(float(acc_obs) - prior))
            post_errs.append(abs(float(acc_obs) - post))
        return float(np.mean(prior_errs)), float(np.mean(post_errs))

    def __len__(self):
        if self.shared:
            return len(self.gp)
        return sum(len(g) for g in self._gps.values())
