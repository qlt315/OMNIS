% Convergence curves from figures/python figures/convergence/plot_data.mat
% Smoothed mean series (Savitzky-Golay); does not save figures.
% Algorithms: OMNIS+ (causal), C-OMNIS+ (CTO), OMNIS (UCB), GDO, PPO, MAPPO.

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(this_dir);
mat_path = fullfile(repo_root, 'figures', 'python figures', 'convergence', 'plot_data.mat');
S = load(mat_path);

algo_keys = {'causal', 'cto', 'ucb', 'gdo', 'ppo', 'mappo'};
algo_labels = {'OMNIS+', 'C-OMNIS+', 'OMNIS', 'GDO', 'PPO', 'MAPPO'};

% Series field -> ylabel (Acc shown in %)
series_fields = {'cum_reward_mean', 'delay_series_mean', 'acc_series_mean', ...
    'energy_series_mean', 'backlog_series_mean', 'vio_series_mean'};
metric_labels = {'Avg. Reward', 'Avg. Latency [s]', 'Avg. Acc. [%]', ...
    'Avg. Energy [J]', 'Avg. Backlog [bits]', 'Avg. Violation Prob.'};
acc_scale = [1, 1, 100, 1, 1, 1];  % Acc -> %
T_max = 500;

% Colors keep original palette order: causal, ucb, gdo, ppo, mappo, cto
palette = lines(6);
[~, cidx] = ismember(algo_keys, {'causal', 'ucb', 'gdo', 'ppo', 'mappo', 'cto'});
colors = palette(cidx, :);

figure('Position', [100, 100, 1100, 700]);
% compact (not none): leave room for bottom xlabels + south legend on resize
tlo = tiledlayout(2, 3, 'Padding', 'compact', 'TileSpacing', 'compact');

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
        T = min(T_max, numel(y));
        y = y(1:T);
        if T >= 49
            y_plot = sgolayfilt(y, 3, 49);
        else
            y_plot = y;
        end
        plot(1:T, y_plot, 'LineWidth', 2.2, 'Color', colors(alg_idx, :), ...
            'DisplayName', algo_labels{alg_idx});
    end

    xlim([1, T_max]);
    if metric_idx > 3
        xlabel('Time Slot', 'FontSize', 14, 'FontName', 'Times New Roman');
    end
    ylabel(metric_labels{metric_idx}, 'FontSize', 14, 'FontName', 'Times New Roman');
    grid on;
    ax.GridColor = [0.2 0.2 0.2];
    ax.GridAlpha = 0.6;
    ax.Box = 'on';
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
end

lgd = legend(algo_labels, 'Orientation', 'horizontal', ...
    'FontSize', 14, 'FontName', 'Times New Roman', ...
    'NumColumns', numel(algo_labels));
lgd.Layout.Tile = 'south';  % reserved band; survives vertical resize
lgd.Box = 'off';
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
% Keep LR compact without killing bottom padding (avoid tighten_lr Padding=none).
tlo.Padding = 'compact';

