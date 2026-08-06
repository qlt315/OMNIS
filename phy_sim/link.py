"""Link-level transport-block (TB) simulation chain: 5G NR LDPC + QAM.

Design decisions
----------------
1. AWGN-conditional tables: the SNR axis of every produced table is the
   per-symbol Es/N0 at the demapper input (channel gain normalized to 1).
   Fast fading is NOT simulated here; it enters the system-level simulation
   through the per-slot SINR traces. This keeps the tables free of any
   channel-model assumptions and makes the SNR calibration contract explicit.
2. Code rates below the LDPC5G minimum mother rate (1/5) — MCS 0, 1, 2 —
   are realized the 38.212 way: a mother code of rate 1/5 whose coded bits
   are circularly repeated; the receiver folds (sums) the LLRs of repeated
   positions before decoding.
3. One task payload = one transport block, segmented into K code blocks of
   k info bits (BG1 if mother rate > 1/4 else BG2, mirroring NR practice).
   TB-level BLER = P(any code block decoded with errors).
"""

import math

import torch
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel import AWGN

MIN_MOTHER_RATE = 0.2
BG1_KMAX = 8448
BG2_KMAX = 3840


def design_tb(payload_bits, mcs):
    """Choose the LDPC code-block layout for one (payload, MCS) pair.

    Returns dict with K (blocks/TB), k (info bits/block), n0 (mother code
    length), bg (base graph), G (transmitted coded bits/block after circular
    repetition, rounded up to a multiple of qm).
    """
    P = payload_bits
    R = mcs.code_rate
    mother_rate = max(R, MIN_MOTHER_RATE)
    # NR base-graph rule: BG2 for mother rates below 1/3, BG1 otherwise
    if mother_rate < 1.0 / 3.0:
        bg, kmax = 'bg2', BG2_KMAX
    else:
        bg, kmax = 'bg1', BG1_KMAX
    K = max(1, math.ceil(P / kmax))
    k = math.ceil(P / K)
    n0 = math.ceil(k / mother_rate)
    G = math.ceil(k / R)
    G = math.ceil(G / mcs.qm) * mcs.qm  # whole modulation symbols
    return dict(K=K, k=k, n0=n0, bg=bg, G=G, mother_rate=mother_rate)


class TBLink:
    """Encode/transmit/decode pipeline for one (payload_bits, mcs) point."""

    def __init__(self, payload_bits, mcs, num_iter=30, device='cpu',
                 precision='single'):
        self.payload_bits = payload_bits
        self.mcs = mcs
        self.device = device
        lay = design_tb(payload_bits, mcs)
        self.layout = lay
        self.K, self.k, self.n0, self.G = lay['K'], lay['k'], lay['n0'], lay['G']
        # num_bits_per_symbol is left unset: the optional code-bit interleaver
        # it enables is not part of NR LDPC processing, and it would constrain
        # n to a multiple of qm, breaking exact mother rates of 1/5.
        self.encoder = LDPC5GEncoder(self.k, self.n0, bg=lay['bg'],
                                     precision=precision, device=device)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=num_iter,
                                     hard_out=True, return_infobits=True,
                                     precision=precision, device=device)
        # 2-PAM is BPSK; needed to reproduce the legacy AWGN+BPSK accuracy data
        const_type = 'pam' if mcs.qm == 1 else 'qam'
        self.mapper = Mapper(const_type, mcs.qm, precision=precision,
                             device=device)
        self.demapper = Demapper('maxlog', const_type, mcs.qm,
                                 precision=precision, device=device)
        self.awgn = AWGN(precision=precision, device=device)

    def _snr_to_no(self, snr_db):
        # Constellations are normalized to unit average symbol energy (Es=1)
        return 10.0 ** (-snr_db / 10.0)

    def encode(self, u):
        """u: (B, k) info bits -> tx coded bits (B, G), circular repetition."""
        c = self.encoder(u)
        if self.G > self.n0:
            reps = math.ceil(self.G / self.n0)
            c = c.repeat(1, reps)[:, :self.G]
        return c

    def channel_llr(self, tx_bits, snr_db):
        """tx_bits: (B, G) -> per-bit LLRs (B, G) through mapping + AWGN."""
        no = self._snr_to_no(snr_db)
        x = self.mapper(tx_bits)
        y = self.awgn(x, no)
        llr = self.demapper(y, no)
        return llr

    def decode(self, llr):
        """Fold repetition LLRs back to mother code length and decode."""
        if self.G > self.n0:
            reps = math.ceil(self.G / self.n0)
            pad = reps * self.n0 - self.G
            if pad > 0:  # zero LLR = erasure, keeps the partial last repetition
                llr = torch.nn.functional.pad(llr, (0, pad))
            llr = llr.reshape(-1, reps, self.n0).sum(dim=1)
        return self.decoder(llr)

    def transmit(self, u, snr_db):
        """u: (B, k) info bits -> decoded info bits (B, k)."""
        return self.decode(self.channel_llr(self.encode(u), snr_db))

    @torch.no_grad()
    def simulate_tb(self, num_tbs, snr_db, batch_tbs=8, generator=None):
        """Monte-Carlo TB simulation with random payloads.

        Returns dict(tb_errors, bit_errors, num_tbs, num_info_bits).
        A TB is in error if ANY of its K code blocks has >= 1 bit error.
        """
        tb_errors = 0
        bit_errors = 0
        done = 0
        while done < num_tbs:
            b = min(batch_tbs, num_tbs - done) * self.K
            u = torch.randint(0, 2, (b, self.k), device=self.device,
                              dtype=torch.float32, generator=generator)
            u_hat = self.transmit(u, snr_db)
            errs = (u_hat != u).sum(dim=1)                      # per code block
            tb_errors += int((errs.reshape(-1, self.K).sum(dim=1) > 0).sum())
            bit_errors += int(errs.sum())
            done += b // self.K
        return dict(tb_errors=tb_errors, bit_errors=bit_errors,
                    num_tbs=num_tbs, num_info_bits=num_tbs * self.K * self.k)


if __name__ == '__main__':
    import sys
    from mcs import get_mcs

    torch.manual_seed(0)
    for mcs_idx in [0, 2, 9, 16, 28]:
        mcs = get_mcs(mcs_idx)
        link = TBLink(payload_bits=48000, mcs=mcs, num_iter=20)
        lay = link.layout
        print(f"MCS {mcs_idx:2d} ({mcs.modulation}, R={mcs.code_rate:.4f}): "
              f"K={lay['K']} k={lay['k']} n0={lay['n0']} G={lay['G']} bg={lay['bg']}")
        for snr in [0.0, 10.0]:
            r = link.simulate_tb(num_tbs=20, snr_db=snr)
            bler = r['tb_errors'] / r['num_tbs']
            ber = r['bit_errors'] / r['num_info_bits']
            print(f"   SNR {snr:5.1f} dB: BLER={bler:.3f} BER={ber:.2e}")
