import numpy as np


class CausalSCM:
    """Structural causal model of the OMNIS task-offloading mechanism.

    Causal graph (mechanisms are invariant across MDs and time slots):
        SNR        --> Accuracy,  SNR --> Rate --> Delay
        Model      --> Accuracy,  Model --> Payload --> {Delay, Energy}
        CodingRate --> Accuracy,  CodingRate --> Payload
        {Accuracy, Delay, Energy} --> Reward   (analytic reward equation)

    Interventions on Model and CodingRate propagate through known mechanisms;
    only the accuracy mechanism P(acc | do(Model, CodingRate), SNR) needs to be
    learned online. The prior mean of that mechanism is built from coarse
    offline measurements (a subsampled SNR grid), emulating pre-deployment
    benchmarking of the DNN branches.
    """

    def __init__(self, models, data_size, available_coding_rate, acc_data, prior_snr_step=10):
        self.models = models
        self.data_size = data_size
        self.available_coding_rate = list(available_coding_rate)

        # Arm factorization a = (quantization method, bottleneck channels)
        self.arm_features = [
            (1.0 if m['quant_method'] == 'Standard' else 0.0, float(m['quant_channel']))
            for m in models
        ]
        self._feature_to_name = {
            feat: m['name'] for feat, m in zip(self.arm_features, models)
        }

        self._acc_prior = self._build_accuracy_prior(acc_data, prior_snr_step)

    def _build_accuracy_prior(self, acc_data, prior_snr_step):
        """Coarse piecewise-linear accuracy curves from a subsampled SNR grid."""
        prior = {}
        for m in self.models:
            for phi in self.available_coding_rate:
                df = acc_data[(acc_data["Model"] == m['name']) & (acc_data["Coding Rate"] == phi)]
                snr = df["SNR"].values[::prior_snr_step]
                acc = df["Accuracy"].values[::prior_snr_step]
                # Always include the grid endpoints so interpolation covers the full range
                if snr[-1] != df["SNR"].values[-1]:
                    snr = np.append(snr, df["SNR"].values[-1])
                    acc = np.append(acc, df["Accuracy"].values[-1])
                prior[(m['name'], phi)] = (snr, acc)
        return prior

    def acc_prior_mean(self, snr_db, model_name, coding_rate):
        """Prior mean of the accuracy mechanism at (do(model, coding_rate), snr_db)."""
        snr, acc = self._acc_prior[(model_name, coding_rate)]
        return float(np.interp(snr_db, snr, acc))

    def arm_feature(self, arm_idx):
        return self.arm_features[arm_idx]

    def feature_to_name(self, quant_flag, channels):
        return self._feature_to_name[(quant_flag, channels)]

    def coded_size(self, model_name, coding_rate):
        """Transmitted bitstream size after channel coding (bytes)."""
        return self.data_size[model_name] / coding_rate
