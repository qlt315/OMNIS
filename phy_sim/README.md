# phy_sim

Offline 5G NR link-level generation for OMNIS. Tables and SINR traces are
consumed by the system simulator; Sionna is not used online.

## Files

| File | Description |
|------|-------------|
| `mcs.py` | 3GPP TS 38.214 Table 5.1.3.1-1 → `output/mcs_def.csv` |
| `payloads.py` | Per-branch payload sizes |
| `link.py` | Transport-block chain: NR LDPC, QAM, AWGN |
| `run_bler.py` | BLER sweep → `output/bler_table.csv` |
| `run_acc.py` | Accuracy sweep → `output/acc_table.csv`, `acc_clean.csv` |
| `run_traces.py` | Multi-cell 38.901 SINR traces |
| `run_cb_scan.py` | Code-block residual-BER scan |
| `synth_acc.py` | Accuracy table from Φ curves without the split DNN |
| `features.py` | Feature / tail interfaces for `run_acc.py` |

## Modeling

1. **AWGN-conditional tables.** The SNR axis is per-symbol Es/N0 at the
   demapper with unit channel gain. Fast fading is not included here; it
   enters the system simulator through the per-slot SINR traces.
2. **MCS 0–2.** Rates below the LDPC mother rate 1/5 follow 38.212:
   rate-1/5 coding with circular repetition and LLR folding at the receiver.
3. **Transport-block BLER.** One task payload is one TB, segmented into K
   code blocks. The TB fails if any code block fails.
4. **Traces.** Intra-cell UEs are orthogonal under bandwidth slicing.
   Other-cell interference is scaled by `1/ues_per_cell`. Traces store all
   cell–UE links; association is decided in the system simulator.
5. **Receive array.** Default BS panel is 2×4; the UE is single-antenna.
   Traces report post-MRC scalar SINR.

## Install

```bash
pip install sionna
```

LDPC decoding dominates runtime. Prefer `--device cuda` for production sweeps.

## Smoke

```bash
python mcs.py
python link.py
python run_bler.py --branches Box3 --mcs 9 16 28 --snr 0 16 2 \
    --num-tbs 100 --round-tbs 50 --batch-tbs 10 --out output/bler_smoke.csv
python run_acc.py --branches Box3 --mcs 9 28 --snr 0 10 5 \
    --num-samples 16 --batch-samples 8 --out output/acc_smoke.csv \
    --clean-out output/acc_clean_smoke.csv
python run_traces.py --cells 3 --ues-per-cell 2 --slots 20 --tag smoke
```

## Production

```bash
python run_bler.py --num-tbs 50000 --target-errors 200 \
    --batch-tbs 64 --device cuda --out output/bler_table.csv

python run_acc.py --provider npz --num-samples 1000 \
    --batch-samples 32 --device cuda --out output/acc_table.csv

python run_traces.py --dataset d1 --cells 7 --ues-per-cell 8 --slots 1000
python run_traces.py --dataset d2 --cells 7 --ues-per-cell 8 --slots 1000 --speed 0.83
```

Sweep scripts append incrementally. Resume by restricting `--branches` / `--mcs`.

## Accuracy without the split DNN

`synth_acc.py` builds `acc_table.csv` from:

1. Branch-wise Φ curves calibrated on `sys_data/acc_data/acc_data.xlsx`
2. Residual BER from `run_cb_scan.py`

```bash
python run_cb_scan.py --mode legacy ...
python run_cb_scan.py --mode mcs ...
python synth_acc.py
```

The system simulator currently uses BLER-gated accuracy from `acc_clean.csv`
and `bler_table.csv` via `omnis/mcs_table.py`.

## Split-DNN accuracy table

`features.py` defines the plug-in interfaces. To generate a real `acc_table.csv`:

1. Export quantized features per branch as `features/<branch>.npz`
   with `payloads` and `labels`.
2. Implement `NPZTailEvaluator.accuracy` around the tail network.
3. Run `python run_acc.py --provider npz`.

`--provider random` is for pipeline smoke tests only.

## Outputs used by OMNIS

| Artifact | Role |
|----------|------|
| `output/mcs_def.csv` | MCS definitions |
| `output/bler_table.csv` | BLER vs model, MCS, SNR |
| `output/acc_clean.csv` | Error-free accuracy anchors |
| `output/sinr_trace_*.csv` | Multi-cell SINR traces |
| `output/link_info_*.csv` | Trace metadata |
