# phy_sim — Sionna-based PHY-layer data generation for OMNIS

Offline link-level data generation for the journal extension. Produces the
lookup tables consumed by the system-level simulation (resource allocation,
queueing, causal MAB). Sionna never enters the online loop — it only produces
CSV tables/traces, per the division of labor in `PHY_DATA_REQUEST.md`.

## Layout

| file | role |
|---|---|
| `mcs.py` | 3GPP TS 38.214 Table 5.1.3.1-1 (MCS 0–28) → `output/mcs_def.csv` |
| `payloads.py` | per-branch payload sizes (mirrors `sys_data/config.py`) |
| `link.py` | TB simulation chain: NR LDPC + QAM + AWGN, with CLI self-test |
| `run_bler.py` | sweep → `bler_table.csv` (model, mcs, snr → BLER) |
| `features.py` | DNN plug-in interfaces + synthetic smoke-test provider |
| `run_acc.py` | sweep → `acc_table.csv` (+ `acc_clean.csv` anchor) |
| `run_traces.py` | multi-cell 38.901 SINR traces (D1 static / D2 mobile) |

## Key design decisions (read before changing anything)

1. **AWGN-conditional tables.** The `snr_db` axis of `acc_table`/`bler_table`
   is per-symbol Es/N0 at the demapper input with channel gain normalized to
   1. Fast fading is *not* simulated here; it enters the system sim through
   the per-slot SINR traces. This keeps tables channel-model-agnostic and
   makes the SNR calibration contract explicit: any trace SINR must be the
   same quantity (post-reception per-symbol SINR).
2. **Rates below 1/5 (MCS 0–2).** LDPC5G cannot go below its mother rate of
   1/5, so these MCS use the 38.212 mechanism: mother code at 1/5 + circular
   repetition; the receiver folds (sums) repeated LLRs before decoding.
3. **Per-branch code-block design.** One task payload = one transport block,
   segmented into K code blocks (bg1 if mother rate > 1/4, else bg2). BLER is
   TB-level: any code block in error ⇒ TB in error. Payloads differ ~7.5×
   across branches, so BLER is genuinely branch-dependent.
4. **Co-slice interference model (traces).** Intra-cell UEs are orthogonal
   (bandwidth slicing); a given slice sees in expectation one UE per other
   cell, implemented as (sum of other-cell powers)/`ues_per_cell`. This keeps
   the SINR distribution scale-invariant in UEs-per-cell.
5. **Serving-link semantics.** Traces record *all* (cell, UE) link SINRs; the
   system sim implements association itself (joint `(model, cell_rank)` bandit
   arm with Top-L pruning, or max-SINR ablation).
6. **Receive-array MIMO (SIMO).** Default BS panel is 2×4 = 8 antennas
   (`--bs-rows/--bs-cols`); UE is single-antenna. Post-MRC scalar SINR
   `P‖h‖²/(N0+I)` is written to the trace — spatial multiplexing is out of
   scope so the AWGN-conditional accuracy/BLER tables stay unchanged.

## Install

```bash
pip install sionna          # 2.0.1; pulls the torch backend
```

Developed on macOS (Apple M2, CPU-only). LDPC decoding is the bottleneck —
production sweeps should run on an NVIDIA GPU server (`--device cuda`).

## Smoke tests (minutes, CPU)

```bash
python mcs.py                                    # mcs_def.csv + table dump
python link.py                                   # chain self-test, 5 MCS × 2 SNR
python run_bler.py --branches Box3 --mcs 9 16 28 --snr 0 16 2 \
    --num-tbs 100 --round-tbs 50 --batch-tbs 10 --out output/bler_smoke.csv
python run_acc.py --branches Box3 --mcs 9 28 --snr 0 10 5 \
    --num-samples 16 --batch-samples 8 --out output/acc_smoke.csv \
    --clean-out output/acc_clean_smoke.csv
python run_traces.py --cells 3 --ues-per-cell 2 --slots 20 --tag smoke
```

## Production runs (GPU server)

```bash
# BLER table: 6 branches × 29 MCS × SNR -5..20 dB (adaptive trials)
python run_bler.py --num-tbs 50000 --target-errors 200 \
    --batch-tbs 64 --device cuda --out output/bler_table.csv

# Accuracy table: requires the real DNN hookup (see below)
python run_acc.py --provider npz --num-samples 1000 \
    --batch-samples 32 --device cuda --out output/acc_table.csv

# Multi-cell traces (fast even on CPU)
python run_traces.py --dataset d1 --cells 7 --ues-per-cell 8 --slots 1000
python run_traces.py --dataset d2 --cells 7 --ues-per-cell 8 --slots 1000 --speed 0.83
```

All sweep scripts append rows incrementally — interrupted runs resume by
re-running with completed (branch, MCS) ranges excluded via `--branches/--mcs`.

## Sanity checks already done

BLER waterfalls vs NR theory (Box3 payload, LDPC 30 iters):

| MCS | modulation, rate | waterfall (measured) | theory |
|---|---|---|---|
| 0–2 | QPSK, 0.12–0.19 | < 0 dB (BLER≈0 at 0 dB) | repetition + heavy coding ✓ |
| 9 | QPSK, 0.663 | 2–4 dB | Shannon 1.6 + gap ✓ |
| 16 | 16QAM, 0.643 | 8–10 dB | Shannon ~7 + gap ✓ |
| 28 | 64QAM, 0.926 | > 16 dB | Shannon ~13.5 + gap ✓ |

Note: these imply accuracy must *collapse* at high MCS even at SNR = 10 dB
(MCS 28 needs ≳ 15 dB) — unlike the externally produced accuracy figure we
started from, whose SNR = 10 dB curves stay flat up to MCS 28. Our tables
resolve that anomaly by construction: single-shot semantics, decode failure
⇒ corrupted payload ⇒ accuracy floor.

## Synthesized accuracy table (no DNN needed) — robustness-curve transplant

`synth_acc.py` builds `acc_table.csv` without the split DNN:

1. **Phi_branch** (accuracy vs payload residual BER) is calibrated from
   `sys_data/acc_data/acc_data.xlsx` (real DNN, AWGN+BPSK). The rate-1.0
   (uncoded) rows give analytic BERs `Q(sqrt(2*snr))` — assumption-free;
   the 5 points collapse onto a clean monotone curve per branch
   (see `output/phi_*.png`).
2. The legacy **coded rows (rates 0.2/0.5) are excluded from Phi by default**:
   fitting an SNR offset to our NR-LDPC reproduction hits the -8 dB search
   boundary (rate 0.2), i.e. the old sim's low-rate code is far weaker than NR
   LDPC (likely under-implemented rate matching / few iterations / additional
   impairments). They are kept as validation markers in the plots. Pass
   `--coded on` to include them anyway.
3. `acc(branch, mcs, snr) = Phi_branch(residual_ber(mcs, snr))`, residual BER
   from `run_cb_scan.py` (isotonic-smoothed). Known approximation:
   post-LDPC-failure errors are bursty, legacy uncoded errors are i.i.d.
4. `bler_table.csv` is assembled analytically: `BLER = 1-(1-p_cb)^K`.

Run order: `run_cb_scan.py --mode legacy` (minutes), `run_cb_scan.py --mode mcs`
(~1-2 h on this machine, two half-jobs in parallel), then `python synth_acc.py`.

## Plugging in the real split DNN (for `acc_table.csv`)

`features.py` defines the two interfaces; the system-level repo does not ship
the DNN. To generate the real table:

1. Export quantized features per branch from the DNN project:
   `features/<branch>.npz` with `payloads: uint8 (N, payload_bytes)` and
   `labels: int64 (N,)`.
2. Implement `NPZTailEvaluator.accuracy`: unpack bits → dequantize → run the
   tail network → return the mean task metric.
3. Run `run_acc.py --provider npz`.

`--provider random` (synthetic logistic accuracy model) exists only to
smoke-test the pipeline.
