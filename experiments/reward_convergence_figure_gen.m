% Reward convergence (single panel) from figures/convergence/plot_data.mat
% Matches Python reward.png style: running-average Lyapunov reward.
% Does not save figures. Plots OMNIS-UCB; omits DQN and OMNIS-TS.

clear; close all; clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
S = load(fullfile(repo_root, 'figures', 'convergence', 'plot_data.mat'));

algo_keys = {'causal', 'ucb', 'gdo', 'rss', 'ppo', 'mappo', 'cto'};
algo_labels = {'OMNIS-Causal', 'OMNIS-UCB', 'GDO', 'RSS', 'PPO', 'MAPPO', 'CTO'};
colors = lines(numel(algo_keys));

win = 60;  % trailing window (same spirit as plot_results.py)

figure('Position', [120, 120, 860, 480]);
ax = gca;
hold on;

ys_for_ylim = [];
for alg_idx = 1:numel(algo_keys)
    key = algo_keys{alg_idx};
    if ~isfield(S.series, key) || ~isfield(S.series.(key), 'rew_series_mean')
        warning('Missing rew_series for %s; skip', key);
        continue;
    end
    entry = S.series.(key);
    y = double(entry.rew_series_mean(:));
    % Trailing-window running mean
    if numel(y) >= win
        y_plot = movmean(y, [win - 1, 0]);
    else
        y_plot = y;
    end
    t = 1:numel(y_plot);

    % Optional seed-std band from raw rew_series if present
    if isfield(entry, 'rew_series')
        raw = double(entry.rew_series);
        if size(raw, 2) == numel(y)
            if size(raw, 1) > 1
                % apply same trailing window per seed, then std
                slid = zeros(size(raw));
                for s = 1:size(raw, 1)
                    if numel(raw(s, :)) >= win
                        slid(s, :) = movmean(raw(s, :), [win - 1, 0]);
                    else
                        slid(s, :) = raw(s, :);
                    end
                end
                mu = mean(slid, 1);
                sd = std(slid, 0, 1);
                fill([t, fliplr(t)], [mu + sd, fliplr(mu - sd)], ...
                    colors(alg_idx, :), 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
                    'HandleVisibility', 'off');
                y_plot = mu(:);
            end
        end
    end

    plot(t, y_plot, 'LineWidth', 2.2, 'Color', colors(alg_idx, :), ...
        'DisplayName', algo_labels{alg_idx});
    t0 = min(50, max(1, floor(numel(y_plot) / 10)));
    ys_for_ylim = [ys_for_ylim; y_plot(t0:end)]; %#ok<AGROW>
end

xlabel('Time Slot', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Running Average Reward', 'FontSize', 14, 'FontName', 'Times New Roman');
title('Per-slot reward (Lyapunov objective)', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';
set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
lgd = legend('Location', 'best', 'FontSize', 12, 'FontName', 'Times New Roman');
lgd.Box = 'off';

if ~isempty(ys_for_ylim)
    lo = min(ys_for_ylim); hi = max(ys_for_ylim);
    pad = max(0.08 * (hi - lo), 0.4);
    ylim([lo - pad, hi + pad]);
end

set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
hold off;
