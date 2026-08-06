"""MCS definitions from 3GPP TS 38.214 Table 5.1.3.1-1 (64QAM, indices 0-28).

Code rates are the exact standard values (R x 1024). This is the same MCS set
used in the accuracy figure produced with the link-level simulator, so the
indices here are the canonical key joining every PHY table in this package.
"""

import csv

# (mcs_index, modulation, qm, code_rate_x1024)
_TABLE_5_1_3_1_1 = [
    (0,  "QPSK",  2, 120),
    (1,  "QPSK",  2, 157),
    (2,  "QPSK",  2, 193),
    (3,  "QPSK",  2, 251),
    (4,  "QPSK",  2, 308),
    (5,  "QPSK",  2, 379),
    (6,  "QPSK",  2, 449),
    (7,  "QPSK",  2, 526),
    (8,  "QPSK",  2, 602),
    (9,  "QPSK",  2, 679),
    (10, "16QAM", 4, 340),
    (11, "16QAM", 4, 378),
    (12, "16QAM", 4, 434),
    (13, "16QAM", 4, 490),
    (14, "16QAM", 4, 553),
    (15, "16QAM", 4, 616),
    (16, "16QAM", 4, 658),
    (17, "64QAM", 6, 438),
    (18, "64QAM", 6, 466),
    (19, "64QAM", 6, 517),
    (20, "64QAM", 6, 567),
    (21, "64QAM", 6, 616),
    (22, "64QAM", 6, 666),
    (23, "64QAM", 6, 719),
    (24, "64QAM", 6, 772),
    (25, "64QAM", 6, 822),
    (26, "64QAM", 6, 873),
    (27, "64QAM", 6, 910),
    (28, "64QAM", 6, 948),
]

MCS_TABLE = "5.1.3.1-1"
NUM_MCS = len(_TABLE_5_1_3_1_1)


class MCS:
    """Spectral parameters of one MCS index."""

    def __init__(self, index, modulation, qm, code_rate_x1024):
        self.index = index
        self.modulation = modulation
        self.qm = qm                      # bits per modulation symbol (log2 M)
        self.code_rate = code_rate_x1024 / 1024.0
        self.spectral_efficiency = self.qm * self.code_rate  # info bits/symbol


MCS_LIST = [MCS(*row) for row in _TABLE_5_1_3_1_1]
MCS_BY_INDEX = {m.index: m for m in MCS_LIST}


def get_mcs(index):
    return MCS_BY_INDEX[index]


def write_mcs_def_csv(path):
    """Write mcs_def.csv per the PHY data request spec."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mcs_index", "modulation", "qm", "code_rate", "spectral_efficiency"])
        for m in MCS_LIST:
            w.writerow([m.index, m.modulation, m.qm,
                        f"{m.code_rate:.6f}", f"{m.spectral_efficiency:.6f}"])
    return path


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(__file__), "output", "mcs_def.csv")
    write_mcs_def_csv(out)
    print(f"wrote {out}")
    for m in MCS_LIST:
        print(f"MCS {m.index:2d}: {m.modulation:5s} qm={m.qm} "
              f"R={m.code_rate:.4f} eta={m.spectral_efficiency:.4f}")
