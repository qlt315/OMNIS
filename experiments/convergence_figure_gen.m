% Convergence curves from figures/convergence/plot_data.mat
% Style aligned with conference OMNIS scripts; does not save figures.
% Algorithms plotted: OMNIS-Causal, OMNIS-UCB (better of UCB/TS), GDO, RSS,
% PPO, MAPPO, CTO. DQN and OMNIS-TS are omitted.

clear; close all; clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
mat_path = fullfile(repo_root, 'figures', 'convergence', 'plot_data.mat');
S = load(mat_path);

% Internal keys in plot_data.mat / display labels
algo_keys = {'causal', 'ucb', 'gdo', 'rss', 'ppo', 'mappo', 'cto'};
algo_labels = {'OMNIS-Causal', 'OMNIS-UCB', 'GDO', 'RSS', 'PPO', 'MAPPO', 'CTO'};

% Series field -> ylabel (Acc shown in %)
series_fields = {'cum_reward_mean', 'delay_series_mean', 'acc_series_mean', ...
    'energy_series_mean', 'backlog_series_mean', 'vio_series_mean'};
metric_labels = {'Avg. Reward', 'Avg. Latency [s]', 'Avg. Acc. [%]', ...
    'Avg. Energy [J]', 'Avg. Backlog [bits]', 'Avg. Violation Prob.'};
acc_scale = [1, 1, 100, 1, 1, 1];  % Acc -> %

colors = lines(numel(algo_keys));
markers = {'-', '-', '-', '-', '-', '-', '-'};

figure('Position', [100, 100, 1100, 700]);
tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

for metric_idx = 1:numel(series_fields)
    ax = nexttile;
    hold on;
    field = series_fields{metric_idx};
    scale = acc_scale(metric_idx);

    for alg_idx = 1:numel(algo_keys)
        key = algo_keys{alg_idx};
        if ~isfield(S.series, key)
            warning('series.%s missing; skip', key);
            continue;
        end
        entry = S.series.(key);
        if ~isfield(entry, field)
            warning('series.%s.%s missing; skip', key, field);
            continue;
        end
        y = double(entry.(field)(:)) * scale;
        % Light smoothing for readability (optional; raw mean still visible)
        if numel(y) >= 49
            y_plot = sgolayfilt(y, 3, 49);
        else
            y_plot = y;
        end
        plot(1:numel(y_plot), y_plot, 'LineWidth', 2.2, ...
            'LineStyle', markers{alg_idx}, 'Color', colors(alg_idx, :), ...
            'DisplayName', algo_labels{alg_idx});
    end

    xlabel('Time Slot', 'FontSize', 14, 'FontName', 'Times New Roman');
    ylabel(metric_labels{metric_idx}, 'FontSize', 14, 'FontName', 'Times New Roman');
    grid on;
    ax.GridColor = [0.2 0.2 0.2];
    ax.GridAlpha = 0.6;
    ax.Box = 'on';
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
end

lgd = legend(algo_labels, 'Orientation', 'horizontal', ...
    'Location', 'southoutside', 'FontSize', 12, 'FontName', 'Times New Roman', ...
    'NumColumns', numel(algo_labels));
lgd.Box = 'off';
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
