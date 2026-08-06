"""Feature-payload providers and tail-model evaluators for the accuracy table.

The accuracy table needs REAL payloads: the quantized intermediate features of
the split DNN, one payload per test sample. This system-level repository does
not contain the DNN itself, so the DNN is plugged in through two interfaces:

- FeatureProvider.payload_matrix(branch) -> (N, P) bit tensor {0,1}
      the quantized features of all N test samples for one branch
- TailEvaluator.accuracy(branch, corrupted_bits) -> float
      task accuracy when the tail model is fed the corrupted reconstructions

`RandomFeatureProvider` + `SyntheticTailEvaluator` implement both with random
payloads and a logistic accuracy-vs-BER model. They exist ONLY to smoke-test
the pipeline end to end; real tables must use the NPZ provider below plus a
tail evaluator wrapping the actual split-DNN tail network.

To produce the real accuracy table:
1. Export quantized features from the split-DNN project, one file per branch:
   features/<branch>.npz with arrays
       payloads : uint8, shape (N, payload_bytes)   (packed bytes)
       labels   : int64, shape (N,)
2. Implement TailEvaluator around the tail network (see NPZTailEvaluator stub)
   and pass its class to run_acc.py --tail.
"""

import os
from abc import ABC, abstractmethod

import numpy as np
import torch

from payloads import PAYLOAD_BYTES

# Placeholder clean accuracies for the synthetic evaluator (smoke test only)
_SYNTH_CLEAN = {
    'Box3': 0.90, 'Box6': 0.92, 'Box12': 0.94,
    'Standard3': 0.92, 'Standard6': 0.94, 'Standard12': 0.95,
}
_SYNTH_FLOOR = 0.10  # random-guess level of the synthetic task


class FeatureProvider(ABC):
    @abstractmethod
    def payload_matrix(self, branch):
        """Return the (N, P) float tensor of payload bits for all samples."""


class TailEvaluator(ABC):
    @abstractmethod
    def accuracy(self, branch, corrupted_bits):
        """corrupted_bits: (N, P) float tensor -> task accuracy (float)."""


class RandomFeatureProvider(FeatureProvider):
    def __init__(self, num_samples=64, seed=0):
        self.num_samples = num_samples
        self.seed = seed

    def payload_matrix(self, branch):
        g = torch.Generator().manual_seed(self.seed + hash(branch) % 1000)
        P = 8 * PAYLOAD_BYTES[branch]
        return torch.randint(0, 2, (self.num_samples, P),
                             generator=g).float()


class SyntheticTailEvaluator(TailEvaluator):
    """Logistic accuracy degradation vs received bit-error fraction."""

    def __init__(self, providers):
        self._providers = providers

    def accuracy(self, branch, corrupted_bits):
        clean = _SYNTH_CLEAN[branch]
        ref = self._providers.payload_matrix(branch)
        ber = (corrupted_bits != ref).float().mean().item()
        return _SYNTH_FLOOR + (clean - _SYNTH_FLOOR) * max(0.0, 1.0 - ber / 0.05) ** 2


class NPZFeatureProvider(FeatureProvider):
    """Loads quantized features exported from the split-DNN project."""

    def __init__(self, directory="features"):
        self.directory = directory

    def payload_matrix(self, branch):
        path = os.path.join(self.directory, f"{branch}.npz")
        data = np.load(path)
        packed = torch.from_numpy(data["payloads"].astype(np.uint8))
        bits = torch.unpackbits(packed.flatten()).reshape(packed.shape[0], -1).float()
        return bits[:, : 8 * PAYLOAD_BYTES[branch]]


class NPZTailEvaluator(TailEvaluator):
    """Wrap the real split-DNN tail network. IMPLEMENT with the DNN project:

    - __init__: load the tail network and labels from the same npz files
    - accuracy: repack corrupted_bits to bytes, dequantize to feature tensors,
      run the tail network batch-wise, return the mean task metric
    """

    def accuracy(self, branch, corrupted_bits):
        raise NotImplementedError(
            "Implement NPZTailEvaluator.accuracy with the real tail network; "
            "see features.py module docstring for the expected npz format.")
