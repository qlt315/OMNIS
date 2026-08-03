import numpy as np
from scipy.special import erf

from omnis.causal_gp import ResidualGP


class CausalMAB:
    """Shared causal contextual bandit for dynamic split-DNN branch selection.

    A single global residual GP learns the accuracy mechanism
    P(acc | do(Model, CodingRate), SNR), which is invariant across MDs
    (mechanism autonomy), so observations from all MDs are pooled. Each MD
    composes the GP-UCB accuracy estimate with its own analytic, context-
    specific QoS penalty terms. Arms are scored by marginalizing the ES
    channel-coding policy -- a downstream, post-action mechanism -- through a
    mechanistic forward simulation, instead of conditioning on its realized
    value (which would induce post-treatment bias).
    """

    def __init__(self, scm, length_scales, signal_var, noise_var, beta,
                 penalty_gain=1.5, acquisition='ucb', use_prior=True, shared=True):
        self.scm = scm
        self.beta = beta
        self.penalty_gain = penalty_gain
        self.acquisition = acquisition
        self.use_prior = use_prior
        self.shared = shared
        self._gp_args = dict(
            prior_mean_fn=self._prior_at,
            length_scales=length_scales,
            signal_var=signal_var,
            noise_var=noise_var,
        )
        # Shared mechanism GP across MDs (mechanism invariance); per-MD GPs are
        # kept only as an ablation of the pooling design.
        self.gp = ResidualGP(**self._gp_args)
        self._gps = {}
        self._pending = {}  # per-MD (snr_db, arm_idx) of the last selected arm

    def _gp_for(self, user):
        if self.shared:
            return self.gp
        if user not in self._gps:
            self._gps[user] = ResidualGP(**self._gp_args)
        return self._gps[user]

    def _make_x(self, snr_db, arm_idx, coding_rate):
        quant_flag, channels = self.scm.arm_feature(arm_idx)
        return np.array([snr_db, quant_flag, channels, coding_rate])

    def _prior_at(self, x):
        if not self.use_prior:
            return 0.0
        snr_db, quant_flag, channels, coding_rate = x
        model_name = self.scm.feature_to_name(quant_flag, channels)
        return self.scm.acc_prior_mean(snr_db, model_name, coding_rate)

    def predict_coding_rate(self, snr_db, model_name, task, predict_overheads):
        """Mechanistic forward simulation of the ES coding-rate policy:
        among feasible rates pick the one with the highest prior accuracy;
        if none is feasible, pick the rate minimizing the penalty term."""
        best_feas, best_feas_acc = None, -np.inf
        best_infeas, best_infeas_pen = None, np.inf
        for phi in self.scm.available_coding_rate:
            delay_hat, energy_hat = predict_overheads(model_name, phi)
            acc_hat = self.scm.acc_prior_mean(snr_db, model_name, phi)
            if delay_hat <= task['delay_constraint'] and energy_hat <= task['energy_constraint']:
                if acc_hat > best_feas_acc:
                    best_feas, best_feas_acc = phi, acc_hat
            else:
                pen = (acc_hat
                       + task['delay_weight'] * erf(delay_hat - task['delay_constraint'])
                       + task['energy_weight'] * erf(energy_hat - task['energy_constraint']))
                if pen < best_infeas_pen:
                    best_infeas, best_infeas_pen = phi, pen
        return best_feas if best_feas is not None else best_infeas

    def select_arm(self, user, snr_db, task, predict_overheads):
        """Score every arm as UCB(acc) + analytic penalty and return the best.

        predict_overheads(model_name, coding_rate) -> (delay_hat, energy_hat)
        is supplied by the simulator and evaluates the analytic causal chain
        Payload -> {Delay, Energy} under a persistence prediction of the ES
        bandwidth/GPU allocation.
        """
        return self.select_arms_batch([(user, snr_db, task, predict_overheads)])[user]

    def select_arms_batch(self, requests):
        """Batched variant of select_arm: one GP posterior solve per slot for
        all MDs' candidate points instead of one solve per MD.

        requests: iterable of (user, snr_db, task, predict_overheads).
        Returns {user: arm_idx}.
        """
        num_arms = len(self.scm.models)
        flat_xs = []
        meta = []
        for user, snr_db, task, predict_overheads in requests:
            phi_hats = []
            overhead_hats = []
            for arm_idx, model in enumerate(self.scm.models):
                phi_hat = self.predict_coding_rate(snr_db, model['name'], task, predict_overheads)
                phi_hats.append(phi_hat)
                overhead_hats.append(predict_overheads(model['name'], phi_hat))
                flat_xs.append(self._make_x(snr_db, arm_idx, phi_hat))
            meta.append((user, task, phi_hats, overhead_hats))

        if self.shared:
            mu, std = self.gp.predict(np.array(flat_xs))
            acc_score = self._acquire(mu, std)
            scores = [acc_score[u_idx * num_arms:(u_idx + 1) * num_arms]
                      for u_idx in range(len(meta))]
        else:
            scores = []
            for u_idx, (user, _, _, _) in enumerate(meta):
                xs_u = np.array(flat_xs[u_idx * num_arms:(u_idx + 1) * num_arms])
                mu, std = self._gp_for(user).predict(xs_u)
                scores.append(self._acquire(mu, std))

        selected = {}
        for u_idx, (user, task, phi_hats, overhead_hats) in enumerate(meta):
            best_val, best_arm = -np.inf, 0
            for arm_idx in range(num_arms):
                delay_hat, energy_hat = overhead_hats[arm_idx]
                val = (scores[u_idx][arm_idx]
                       + self.penalty_gain * task['delay_weight']
                       * erf(task['delay_constraint'] - delay_hat)
                       + self.penalty_gain * task['energy_weight']
                       * erf(task['energy_constraint'] - energy_hat))
                if val > best_val:
                    best_val, best_arm = val, arm_idx
            selected[user] = best_arm
            snr_db = flat_xs[u_idx * num_arms][0]
            self._pending[user] = (snr_db, best_arm)
        return selected

    def _acquire(self, mu, std):
        if self.acquisition == 'ts':
            return mu + std * np.random.randn(len(mu))
        return mu + self.beta * std

    def register_outcome(self, user, coding_rate_realized, acc_obs):
        """Register the realized interventional observation in the GP."""
        snr_db, arm_idx = self._pending.pop(user)
        self._gp_for(user).add(self._make_x(snr_db, arm_idx, coding_rate_realized), acc_obs)

    def register_outcomes_batch(self, records):
        """Register one slot of realized outcomes: [(user, coding_rate, acc_obs)]."""
        for user, coding_rate_realized, acc_obs in records:
            self.register_outcome(user, coding_rate_realized, acc_obs)

    def __len__(self):
        if self.shared:
            return len(self.gp)
        return sum(len(g) for g in self._gps.values())
