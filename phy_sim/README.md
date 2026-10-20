# phy_sim

Sionna-based PHY for OMNIS+: MCS tables, TB BLER / accuracy sweeps, and multi-cell
channel traces. The system simulator loads results through `omnis/mcs_table.py`
and `omnis/channel_trace.py`.

Payload sizes match the six branches in `sys_data/config.py`
(Box3/6/12, Standard3/6/12). One inference task is one transport block.

## Pipeline

```
mcs.py        →  mcs_def.csv
link.py       →  used by run_bler / run_acc
run_bler.py   →  bler_table.csv
run_acc.py    →  acc_table.csv, acc_clean.csv
run_traces.py →  channel_<tag>.npz
```

## Files

| File | Description |
|------|-------------|
| `mcs.py` | PUSCH MCS Table 5.1.3.1-1 (no transform precoding; indices 0–28) |
| `payloads.py` | Branch payload sizes (keep in sync with `Config.data_size`) |
| `link.py` | Encode / transmit / decode one transport block |
| `features.py` | Load feature payloads and score accuracy with the split-DNN tail |
| `run_bler.py` | Sweep (branch, MCS, SNR) → TB BLER |
| `run_acc.py` | Sweep (branch, MCS, SNR) → task accuracy through the link + tail |
| `run_traces.py` | Generate 7-cell UMi uplink channel traces |

## Install

```bash
pip install sionna
```

Sionna 2.x with a PyTorch backend. Use `--device cuda` for large sweeps.

## How to run

```bash
cd phy_sim

# MCS definitions (TS 38.214 Table 5.1.3.1-1)
python mcs.py

# BLER table (smoke / full)
python run_bler.py --branches Box3 --mcs 9 16 28 --snr 0 16 2 \
    --num-tbs 100 --round-tbs 50 --batch-tbs 10 --out output/bler_smoke.csv
python run_bler.py --num-tbs 50000 --target-errors 200 \
    --batch-tbs 64 --device cuda --out output/bler_table.csv

# Accuracy table (feature NPZ + split-DNN tail)
python run_acc.py --provider npz --num-samples 1000 \
    --batch-samples 32 --device cuda --out output/acc_table.csv

# Channel traces
python run_traces.py --slots 1000 --tag smoke7
```

`link.py` can be run alone for a short self-test.

## Payloads

| Branch | Bytes |
|--------|------:|
| Box3 / Box6 / Box12 | 6000 / 13260 / 33580 |
| Standard3 / Standard6 / Standard12 | 11230 / 22460 / 44930 |

## Link (`link.py`)

`TBLink` sends one TB at a given payload and MCS.
Tables use AWGN Es/N0 at the demapper; time-varying SINR comes from the traces.
Code-block layout follows 5G NR (BG1/BG2, rate matching). A TB fails if any
code block has an information-bit error after LDPC decoding.

## BLER (`run_bler.py`)

Monte Carlo TB BLER over branch × MCS × SNR.
Default grid: MCS 0–28, SNR −5…20 dB (1 dB step).
Each point stops after enough TB errors or a trial cap; CSV rows append so runs
can resume.

## Accuracy (`run_acc.py`)

Sends real feature payloads through `TBLink`, then scores the recovered bits
with the split-DNN tail (`--provider npz`, features under `features/<branch>.npz`).
Writes `acc_table.csv` and clean (error-free) accuracy in `acc_clean.csv`.

## Channel traces (`run_traces.py`)

Uplink MIMO channels under 3GPP TR 38.901 UMi: 7 cells, ISD 200 m,
BS 2×4 antennas, MD 2 antennas, 0.8 m/s mobility, 1 s slots.
Stores `H[slot, cell, ue, n_rx, n_tx]` in `channel_<tag>.npz`.
At runtime the simulator computes IRC SINR and EESM; SINR is not stored in the file.

```bash
python run_traces.py --slots 1000 --tag smoke7
```

## Outputs used by OMNIS+

| File | Used for |
|------|----------|
| `output/mcs_def.csv` | MCS indices and rates |
| `output/bler_table.csv` | Goodput / delay |
| `output/acc_table.csv` | Realized task accuracy |
| `output/acc_clean.csv` | Clean accuracy per branch |
| `output/channel_*.npz` | Multi-cell channel gains |
| `output/channel_*_meta.json` | Antenna / power-control metadata |

Split-DNN training lives in `../split_dnn/` (not called at runtime).
