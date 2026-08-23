# Sweep STATUS

Updated: 2026-08-23T20:45:08

## Finished

- `snr` → `figures/sweeps/snr/`
- `users` → `figures/sweeps/users/`
- `arrival` → `figures/sweeps/arrival/`
- `action_pick` → `figures/sweeps/action_pick/`

## Notes

- algos=['causal', 'ucb', 'dts', 'gdo', 'rss', 'dqn', 'ppo', 'mappo', 'cto'] seeds=(0, 1, 2) slots=500 users=10
- users axis=[5, 10, 15, 20, 25]; UE pool=25 (stable prefix); TRACE_MEAN_BEST_CELL_SINR_DB≈3.11 (calibrated on 25-UE pool)
- sinr_offset_db = snr_target_db - TRACE_MEAN_BEST_CELL_SINR_DB; axis = mean best-cell / serving SINR (not all-cell mean)
- plots: mean lines only (no error bars); log-y for delay/backlog; causal_drift_gain=1.0; GDO=online emp Acc-floor; Acc table env-only
- total wall 7.55 h
- snr artifacts: acc_vs_snr_db.png, action_pick_all.png, action_pick_snr_db0.png, action_pick_snr_db10.png, action_pick_snr_db2.png, action_pick_snr_db4.png, action_pick_snr_db6.png, action_pick_snr_db8.png, backlog_vs_snr_db.png, delay_vs_snr_db.png, energy_vs_snr_db.png, perseed.csv, reward_vs_snr_db.png, snr.mat, snr.npz, snr.pkl, vio_vs_snr_db.png
- users artifacts: acc_vs_n_users.png, action_pick_all.png, action_pick_n_users10.png, action_pick_n_users15.png, action_pick_n_users20.png, action_pick_n_users25.png, action_pick_n_users5.png, backlog_vs_n_users.png, delay_vs_n_users.png, energy_vs_n_users.png, perseed.csv, reward_vs_n_users.png, users.mat, users.npz, users.pkl, vio_vs_n_users.png
- arrival artifacts: acc_vs_arrival_rate.png, arrival.mat, arrival.npz, arrival.pkl, backlog_vs_arrival_rate.png, delay_vs_arrival_rate.png, energy_vs_arrival_rate.png, perseed.csv, reward_vs_arrival_rate.png, vio_vs_arrival_rate.png
- action_pick artifacts: acc_vs_explore_knob.png, acc_vs_n_users.png, acc_vs_snr_db.png, action_pick_explore.mat, action_pick_explore.npz, action_pick_explore.pkl, action_pick_snr.mat, action_pick_snr.npz, action_pick_snr.pkl, action_pick_snr_metrics.mat, action_pick_snr_metrics.npz, action_pick_snr_metrics.pkl, action_pick_users.mat, action_pick_users.npz, action_pick_users.pkl, action_pick_users_metrics.mat, action_pick_users_metrics.npz, action_pick_users_metrics.pkl, action_pick_vs_snr.png, action_pick_vs_users.png, backlog_vs_explore_knob.png, backlog_vs_n_users.png, backlog_vs_snr_db.png, delay_vs_explore_knob.png, delay_vs_n_users.png, delay_vs_snr_db.png, energy_vs_explore_knob.png, energy_vs_n_users.png, energy_vs_snr_db.png, perseed_explore.csv, perseed_snr.csv, perseed_users.csv, reward_vs_explore_knob.png, reward_vs_n_users.png, reward_vs_snr_db.png, vio_vs_explore_knob.png, vio_vs_n_users.png, vio_vs_snr_db.png
