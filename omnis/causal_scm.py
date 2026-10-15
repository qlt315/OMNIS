import numpy as np


class CausalSCM:
    """Mechanisms for (model, SINR, MCS) → accuracy / delay / energy.

    Accuracy P(acc | do(model), SINR, MCS rate) is learned online (GP).
    Delay and energy use the wireless model and measurement-driven stats.
    """

    def __init__(self, models, data_size, mcs_table, prior_snr_step=10,
                 build_prior=False):
        self.models = models
        self.data_size = data_size  # bytes; bits = 8×
        self.mcs_table = mcs_table
        self.available_mcs = list(mcs_table.mcs_indices)

        self.arm_features = [
            (1.0 if m['quant_method'] == 'Standard' else 0.0, float(m['quant_channel']))
            for m in models
        ]
        self._feature_to_name = {
            feat: m['name'] for feat, m in zip(self.arm_features, models)
        }

        del prior_snr_step
        if build_prior:
            raise ValueError(
                "build_prior is unsupported; accuracy prior is zero-mean GP")
        self._acc_prior = {}

    def acc_prior_mean(self, snr_db, model_name, mcs_idx):
        if not self._acc_prior:
            return 0.0
        snr, acc = self._acc_prior[(model_name, mcs_idx)]
        return float(np.interp(snr_db, snr, acc))

    def goodput_se(self, model_name, mcs_idx, snr_db):
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
        return 8.0 * float(self.data_size[model_name])

    def se(self, mcs_idx):
        return self.mcs_table.se[mcs_idx]
