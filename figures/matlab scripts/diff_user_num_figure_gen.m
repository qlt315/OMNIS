% User-count sweep from figures/python figures/sweeps/users/users.mat
% Style aligned with conference OMNIS scripts; does not save figures.
% Plots OMNIS (better of UCB/TS); omits DQN and OMNIS-TS.

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(this_dir);
data = load(fullfile(repo_root, 'figures', 'python figures', 'sweeps', 'users', 'users.mat'));

user_counts = double(data.axis(:)');
algo_keys = {'causal', 'cto', 'ucb', 'gdo', 'ppo', 'mappo'};
algo_labels = {'OMNIS+', 'C-OMNIS+', 'OMNIS', 'GDO', 'PPO', 'MAPPO'};

metric_keys = {'reward', 'delay', 'energy', 'acc', 'vio', 'backlog'};
metric_labels = {'Avg. Reward', 'Avg. Latency [s]', 'Avg. Energy [J]', ...
    'Avg. Acc. [%]', 'Avg. Violation Prob.', 'Avg. Backlog [bits]'};
acc_scale = [1, 1, 1, 100, 1, 1];

% Colors/markers keep original palette order: causal, ucb, gdo, ppo, mappo, cto
palette = lines(6);
[~, cidx] = ismember(algo_keys, {'causal', 'ucb', 'gdo', 'ppo', 'mappo', 'cto'});
colors = palette(cidx, :);
base_markers = {'o', 's', 'd', '^', 'v', 'p'};
markers = base_markers(cidx);

figure('Position', [100, 100, 1100, 520]);
tiledlayout(2, 3, 'Padding', 'none', 'TileSpacing', 'tight');

for metric_idx = 1:numel(metric_keys)
    ax = nexttile;
    hold on;
    mkey = metric_keys{metric_idx};
    scale = acc_scale(metric_idx);

    for alg_idx = 1:numel(algo_keys)
        fname = sprintf('%s_%s_mean', algo_keys{alg_idx}, mkey);
        if ~isfield(data, fname)
            warning('Field "%s" not found; skip', fname);
            continue;
        end
        y = double(data.(fname)(:)') * scale;
        plot(user_counts, y, '-', 'LineWidth', 2.5, ...
            'Color', colors(alg_idx, :), 'Marker', markers{alg_idx}, ...
            'DisplayName', algo_labels{alg_idx}, 'MarkerSize', 8);
    end

    if metric_idx > 3
        xlabel('Number of MDs', 'FontSize', 14, 'FontName', 'Times New Roman');
    end
    ylabel(metric_labels{metric_idx}, 'FontSize', 14, 'FontName', 'Times New Roman');
    xticks(user_counts);
    xlim([min(user_counts), max(user_counts)]);
    grid on;
    ax.GridColor = [0.2 0.2 0.2];
    ax.GridAlpha = 0.6;
    ax.Box = 'on';
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
end

lgd = legend(algo_labels, 'Orientation', 'horizontal', ...
    'FontSize', 14, 'FontName', 'Times New Roman', 'NumColumns', numel(algo_labels));
lgd.Layout.Tile = 'south';
lgd.Box = 'off';
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
tighten_lr(gcf);
