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
    accuracy mechanism P(acc | do(Model, MCS), SINR) is learned online from
    observations — the Acc table is environment-only and must not seed a prior.
    """

    def __init__(self, models, data_size, mcs_table, prior_snr_step=10,
                 build_prior=False):
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

        # Acc table is env-only: never build a decision-time prior from it.
        # Empty / uninformative prior (mean 0); residual GP learns from obs.
        del prior_snr_step  # unused; kept for call-site compatibility
        if build_prior:
            raise ValueError(
                "Building Acc prior from mcs_table.accuracy is banned; "
                "Acc table is environment-only (realize_accuracy / get_accuracy).")
        self._acc_prior = {}

    def acc_prior_mean(self, snr_db, model_name, mcs_idx):
        """Uninformative prior mean (0). Table Acc is never used here."""
        if not self._acc_prior:
            return 0.0
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
