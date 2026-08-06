import numpy as np


class CausalSCM:
    """Structural causal model of the OMNIS task-offloading mechanism.

    Causal graph (mechanisms are invariant across MDs and time slots):
        Cell    --> SINR    --> Accuracy,  SINR --> achievable MCS --> Delay
        Model   --> Accuracy,  Model --> Payload --> {Delay, Energy}
        MCS     --> Accuracy,  MCS --> spectral efficiency --> {Delay, Energy}
        {Accuracy, Delay, Energy} --> Reward   (analytic reward equation)

    Cell is not part of the arm feature factorization; it enters through the
    realized SINR at association time. Interventions on Model propagate through
    known mechanisms; the ES MCS policy is a downstream, post-action mechanism
    that the bandit marginalizes by mechanistic forward simulation. Only the
    accuracy mechanism P(acc | do(Model, MCS), SINR) is learned online.
    """

    def __init__(self, models, data_size, mcs_table, prior_snr_step=10):
        self.models = models
        self.data_size = data_size  # payload size per model [bytes]
        self.mcs_table = mcs_table
        self.available_mcs = list(mcs_table.mcs_indices)

        # Arm factorization a = (quantization method, bottleneck channels)
        self.arm_features = [
            (1.0 if m['quant_method'] == 'Standard' else 0.0, float(m['quant_channel']))
            for m in models
        ]
        self._feature_to_name = {
            feat: m['name'] for feat, m in zip(self.arm_features, models)
        }

        self._acc_prior = self._build_accuracy_prior(mcs_table, prior_snr_step)

    def _build_accuracy_prior(self, mcs_table, prior_snr_step):
        """Coarse piecewise-linear BLER-gated accuracy curves from the BLER grid."""
        prior = {}
        for m in self.models:
            for mcs in self.available_mcs:
                snr_grid, _ = mcs_table._bler[m['name']][mcs]
                snr = snr_grid[::prior_snr_step]
                if snr[-1] != snr_grid[-1]:
                    snr = np.append(snr, snr_grid[-1])
                acc = np.array([mcs_table.accuracy(m['name'], mcs, s) for s in snr])
                prior[(m['name'], mcs)] = (snr, acc)
        return prior

    def acc_prior_mean(self, snr_db, model_name, mcs_idx):
        """Prior mean of the accuracy mechanism at (do(model, mcs), snr_db)."""
        snr, acc = self._acc_prior[(model_name, mcs_idx)]
        return float(np.interp(snr_db, snr, acc))

    def goodput_se(self, model_name, mcs_idx, snr_db):
        """Effective SE after TB erasures."""
        return self.mcs_table.goodput_se(model_name, mcs_idx, snr_db)

    def bler(self, model_name, mcs_idx, snr_db):
        return self.mcs_table.bler(model_name, mcs_idx, snr_db)

    def bler_target(self):
        return self.mcs_table.bler_target

    def arm_feature(self, arm_idx):
        return self.arm_features[arm_idx]

    def feature_to_name(self, quant_flag, channels):
        return self._feature_to_name[(quant_flag, channels)]

    def payload_bits(self, model_name):
        """Transmitted payload size; the MCS code rate acts on spectral
        efficiency, not on the on-air bit count. Keeps the conference-version
        convention that data_size is the on-air size."""
        return self.data_size[model_name]

    def se(self, mcs_idx):
        """Spectral efficiency [bit/s/Hz] of the given MCS."""
        return self.mcs_table.se[mcs_idx]
