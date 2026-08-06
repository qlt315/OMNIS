# phy_sim

This package generates the physical-layer inputs used by OMNIS: MCS
definitions, transport-block BLER tables, accuracy tables, and multi-cell
SINR traces. All link simulations follow 5G NR procedures in Sionna
(LDPC5G, QAM mapping, 3GPP 38.901 channels). The system simulator reads
the CSV outputs through `omnis/mcs_table.py` and `omnis/sinr_trace.py`.

Payload sizes match the six split-DNN branches in `sys_data/config.py`
(Box3/6/12, Standard3/6/12). One inference task is one transport block.

## Overview of the pipeline

```
mcs.py ──────────────► mcs_def.csv
payloads.py ──┐
link.py ──────┼──► run_bler.py ──► bler_table.csv
              ├──► run_acc.py  ──► acc_table.csv, acc_clean.csv
              └──► run_cb_scan.py ──► cb_scan.csv ──► synth_acc.py
run_traces.py ────────────────────► sinr_trace_*.csv, link_info_*.csv
```

Typical workflow: build MCS definitions, sweep BLER and accuracy over
`(branch, MCS, SNR)`, generate SINR traces for the multi-cell scenario,
then point `sys_data/config.py` at the resulting files under `output/`.

## Install

```bash
pip install sionna
```

Developed with Sionna 2.x and a PyTorch backend. LDPC decoding is the
main cost; use `--device cuda` for large sweeps.

---

## MCS definitions (`mcs.py`)

Implements 3GPP TS 38.214 Table 5.1.3.1-1 (64QAM table, indices 0–28).
Each index stores modulation order `qm`, code rate \(R = R_{1024}/1024\),
and spectral efficiency \(qm \cdot R\).

```bash
python mcs.py   # writes output/mcs_def.csv
```

Columns: `mcs_index, modulation, qm, code_rate, spectral_efficiency`.

## Payloads (`payloads.py`)

| Branch | Bytes | Bits |
|--------|------:|-----:|
| Box3 | 6000 | 48000 |
| Box6 | 13260 | 106080 |
| Box12 | 33580 | 268640 |
| Standard3 | 11230 | 89840 |
| Standard6 | 22460 | 179680 |
| Standard12 | 44930 | 359440 |

Keep these values aligned with `Config.data_size` in `sys_data/config.py`.

## Link chain (`link.py`)

`TBLink` encodes, transmits, and decodes one transport block for a given
payload length and MCS.

**SNR definition.** Tables are conditioned on AWGN Es/N0 at the demapper
input with unit average symbol energy. Fast fading is not applied in the
link chain; time-varying SINR is provided separately by the traces.

**Code-block layout (`design_tb`).**

- Mother rate \(\max(R, 0.2)\). Rates below 1/5 use a rate-1/5 mother code
  with circular repetition of coded bits; the receiver folds LLRs before
  decoding (38.212).
- Base graph: BG2 if mother rate \(< 1/3\), else BG1. Maximum info bits
  per block: 3840 (BG2) or 8448 (BG1).
- Number of code blocks \(K = \lceil P / k_{\max}\rceil\); info bits per
  block \(k = \lceil P / K\rceil\).
- Transmitted coded bits per block \(G\) are rounded up to a multiple of
  `qm`.

**Decoder.** Sionna `LDPC5GDecoder` with hard info-bit output. Default
`num_iter=30`. Mapper/demapper use max-log LLRs; QAM for `qm≥2`, PAM/BPSK
for `qm=1` (legacy calibration).

**TB error event.** A transport block is wrong if any of its \(K\) code
blocks has at least one information-bit error after decoding.

```bash
python link.py   # self-test over a few MCS × SNR points
```

---

## BLER table (`run_bler.py`)

Monte Carlo sweep of TB-level BLER over branch, MCS, and SNR.

**Output.** `model,mcs_index,snr_db,bler,n_trials`

**Default grid.**

| Argument | Default |
|----------|---------|
| `--branches` | all six |
| `--mcs` | 0 … 28 |
| `--snr` | −5 … 20 dB, step 1 |
| `--num-tbs` | 20000 max TBs per grid point |
| `--round-tbs` | 1000 TBs per adaptive round |
| `--target-errors` | stop after 100 TB errors |
| `--batch-tbs` | 8 parallel TBs |
| `--num-iter` | 30 LDPC iterations |
| `--device` | `cpu` |
| `--seed` | 1000 |

Adaptive stopping concentrates samples near the waterfall: each point
runs until `target-errors` TB failures or `num-tbs` trials. Rows append
to the CSV so interrupted runs can resume by narrowing `--branches` /
`--mcs`.

```bash
# smoke
python run_bler.py --branches Box3 --mcs 9 16 28 --snr 0 16 2 \
    --num-tbs 100 --round-tbs 50 --batch-tbs 10 --out output/bler_smoke.csv

# production
python run_bler.py --num-tbs 50000 --target-errors 200 \
    --batch-tbs 64 --device cuda --out output/bler_table.csv
```

## Accuracy table (`run_acc.py`)

Transmits real or synthetic feature payloads through `TBLink`, then scores
the reconstructed bits with a tail evaluator.

**Outputs.**

- `acc_table.csv`: `model,mcs_index,snr_db,accuracy,n_trials`
- `acc_clean.csv`: error-free accuracy per branch (written once)

**Providers.**

- `--provider random`: random bit payloads + logistic synthetic accuracy
  for pipeline checks.
- `--provider npz`: load `features/<branch>.npz` with arrays
  `payloads` (uint8, packed) and `labels` (int64); implement
  `NPZTailEvaluator.accuracy` around the split-DNN tail.

**Default grid.** Same SNR/MCS ranges as BLER; `--num-samples` defaults
to 64 (use ~1000 for production), `--batch-samples` 16, `--seed` 2000.

```bash
python run_acc.py --branches Box3 --mcs 9 28 --snr 0 10 5 \
    --num-samples 16 --batch-samples 8 \
    --out output/acc_smoke.csv --clean-out output/acc_clean_smoke.csv

python run_acc.py --provider npz --num-samples 1000 \
    --batch-samples 32 --device cuda --out output/acc_table.csv
```

OMNIS currently forms operating accuracy as
\((1-\mathrm{BLER})\cdot\mathrm{acc\_clean} + \mathrm{BLER}\cdot\mathrm{floor}\)
in `omnis/mcs_table.py`, using `bler_table.csv` and `acc_clean.csv`.

## Code-block scan (`run_cb_scan.py`)

Cheaper CB-level statistics used by `synth_acc.py`. Because TB BLER
satisfies \(\mathrm{BLER}_{TB}=1-(1-p_{CB})^K\), scanning one reference
code block per MCS is enough for table assembly.

**Modes.**

- `--mode mcs`: NR MCS 0–28 at reference \(k\) (BG1: 8000, BG2: 3693).
- `--mode legacy`: LDPC at rates 0.2 and 0.5 with BPSK, matching the
  PHY used in `sys_data/acc_data/acc_data.xlsx`.

**Output.** `config,mcs_index,qm,code_rate,bg,k,snr_db,cb_fail_rate,residual_ber,n_trials`

```bash
python run_cb_scan.py --mode legacy ...
python run_cb_scan.py --mode mcs --snr -5 20 1 ...
```

## Synthesized accuracy (`synth_acc.py`)

Builds `acc_table.csv` without exporting split-DNN features:

1. Fit branch-wise Φ curves (accuracy vs residual BER) from
   `acc_data.xlsx`, primarily using uncoded rate-1.0 rows with analytic
   BER \(Q(\sqrt{2\cdot\mathrm{SNR}})\).
2. Map each `(MCS, SNR)` residual BER from `run_cb_scan.py` through Φ.
3. Optionally assemble `bler_table.csv` via \(1-(1-p_{CB})^K\) with the
   system payload sizes.

```bash
python synth_acc.py --calib-only   # Φ plots only
python synth_acc.py                # full tables
```

---

## SINR traces (`run_traces.py`)

Generates multi-cell uplink SINR under 3GPP TR 38.901 using Sionna’s
hexagonal topology with wraparound.

**Channel.** Default UMi at 3.5 GHz, path loss and shadow fading on.
BS panel: `--bs-rows` × `--bs-cols` (default 2×4, single-pol V,
38.901 pattern). UE: single-antenna omni. Direction: uplink.
Transmit power 0.1 W, bandwidth \(10^5\) Hz, noise from −174 dBm/Hz.
Slot duration 1 ms; one sample per slot.

**Topology.** `gen_hexgrid_topology` with `--num-rings` (default 1 →
7 sites × 3 sectors = 21 sector BSs). Each sector has
`--ues-per-cell` UEs. Inter-site distance `--isd` (default 200 m).
UE speed `--speed` (m/s); 0 for static.

**Interference.** Frequency reuse `--reuse` (default 3). Only
co-channel sectors interfere. Intra-sector UEs are treated as orthogonal
under bandwidth slicing. Interference is instantaneous per slot.

**SINR.** Post-MRC scalar
\(P\|\mathbf{h}\|^2 / (N_0 + I)\) after summing receive-antenna and path
powers. Site-level traces take the max over the three co-sited sectors
so that `num_sites` logical cells match the system model.

**Outputs** (tag = `--tag`, default `smoke7`):

| File | Columns |
|------|---------|
| `sinr_trace_<tag>.csv` | sector-level `slot,cell_id,ue_id,sinr_db` |
| `link_info_<tag>.csv` | sector distances |
| `sinr_trace_<tag>_sites.csv` | site-level SINR for the system simulator |
| `link_info_<tag>_sites.csv` | site-level distances |

```bash
# smoke
python run_traces.py --cells 3 --ues-per-cell 2 --slots 20 --tag smoke

# static / mobile campaigns
python run_traces.py --scenario umi --num-rings 1 --ues-per-cell 8 \
    --slots 1000 --reuse 3 --tag smoke7_sites
python run_traces.py --scenario umi --num-rings 1 --ues-per-cell 8 \
    --slots 1000 --speed 0.83 --tag d2
```

Point `config.sinr_trace_dir` / `sinr_trace_tag` at the site-level files.

---

## Outputs consumed by OMNIS

| Artifact | Consumer |
|----------|----------|
| `output/mcs_def.csv` | `McsTable` |
| `output/bler_table.csv` | BLER / goodput |
| `output/acc_clean.csv` | Clean accuracy anchors |
| `output/sinr_trace_*_sites.csv` | `SinrTrace` Top-L association |
| `output/link_info_*_sites.csv` | Trace metadata |
