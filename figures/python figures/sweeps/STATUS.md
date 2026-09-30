# Sweep STATUS

Updated: 2026-09-30T14:28:17

## Finished

- `snr` → `figures/python figures/sweeps/snr/`
- `users` → `figures/python figures/sweeps/users/`
- `arrival` → `figures/python figures/sweeps/arrival/`
- `action_pick` → `figures/python figures/sweeps/action_pick/`

## Notes

- algos=['causal', 'ucb', 'dts', 'gdo', 'dqn', 'ppo', 'mappo', 'cto'] seeds=(0, 1, 2) slots=500 users=10
- users axis=[5, 10, 15, 20, 25]; UE pool=25; TRACE_MEAN_BEST_CELL_SINR_DB≈6.61
- total wall 1.49 h
- snr artifacts: acc_vs_snr_db.png, backlog_vs_snr_db.png, delay_vs_snr_db.png, energy_vs_snr_db.png, perseed.csv, reward_vs_snr_db.png, snr.mat, snr.npz, snr.pkl, vio_vs_snr_db.png
- users artifacts: acc_vs_n_users.png, backlog_vs_n_users.png, delay_vs_n_users.png, energy_vs_n_users.png, perseed.csv, reward_vs_n_users.png, users.mat, users.npz, users.pkl, vio_vs_n_users.png
- arrival artifacts: acc_vs_arrival_rate.png, arrival.mat, arrival.npz, arrival.pkl, backlog_vs_arrival_rate.png, delay_vs_arrival_rate.png, energy_vs_arrival_rate.png, perseed.csv, reward_vs_arrival_rate.png, vio_vs_arrival_rate.png
- action_pick artifacts: action_pick_snr.mat, action_pick_snr.npz, action_pick_snr.pkl, action_pick_users.mat, action_pick_users.npz, action_pick_users.pkl, action_pick_vs_snr.png, action_pick_vs_users.png, perseed_snr.csv, perseed_users.csv
