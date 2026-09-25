% Reward convergence (single panel) from figures/python figures/convergence/plot_data.mat
% Matches Python reward.png style: running-average Lyapunov reward.
% Does not save figures. Plots OMNIS; omits DQN and OMNIS-TS.

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(this_dir);
S = load(fullfile(repo_root, 'figures', 'python figures', 'convergence', 'plot_data.mat'));

algo_keys = {'causal', 'cto', 'ucb', 'gdo', 'ppo', 'mappo'};
algo_labels = {'OMNIS+', 'C-OMNIS+', 'OMNIS', 'GDO', 'PPO', 'MAPPO'};
% Colors keep original palette order: causal, ucb, gdo, ppo, mappo, cto
palette = lines(6);
[~, cidx] = ismember(algo_keys, {'causal', 'ucb', 'gdo', 'ppo', 'mappo', 'cto'});
colors = palette(cidx, :);

win = 60;  % trailing window (same spirit as plot_results.py)
T_max = 500;

figure('Position', [120, 120, 860, 480]);
ax = gca;
hold on;

ys_for_ylim = [];
for alg_idx = 1:numel(algo_keys)
    key = algo_keys{alg_idx};
    if ~isfield(S.series, key) || ~isfield(S.series.(key), 'rew_series')
        warning('Missing rew_series for %s; skip', key);
        continue;
    end
    entry = S.series.(key);
    raw = double(entry.rew_series);
    T = min(T_max, size(raw, 2));
    raw = raw(:, 1:T);

    slid = zeros(size(raw));
    for s = 1:size(raw, 1)
        if T >= win
            slid(s, :) = movmean(raw(s, :), [win - 1, 0]);
        else
            slid(s, :) = raw(s, :);
        end
    end
    mu = mean(slid, 1);
    t = 1:T;
    if size(slid, 1) > 1
        sd = std(slid, 0, 1);
        fill([t, fliplr(t)], [mu + sd, fliplr(mu - sd)], ...
            colors(alg_idx, :), 'FaceAlpha', 0.12, 'EdgeColor', 'none', ...
            'HandleVisibility', 'off');
    end

    plot(t, mu, 'LineWidth', 2.2, 'Color', colors(alg_idx, :), ...
        'DisplayName', algo_labels{alg_idx});
    t0 = min(50, max(1, floor(T / 10)));
    ys_for_ylim = [ys_for_ylim; mu(t0:end)']; %#ok<AGROW>
end

xlim([1, T_max]);
xlabel('Time Slot', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Average Reward', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';
set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
lgd = legend('Location', 'best', 'FontSize', 14, 'FontName', 'Times New Roman');
lgd.Box = 'off';

if ~isempty(ys_for_ylim)
    lo = min(ys_for_ylim); hi = max(ys_for_ylim);
    pad = max(0.08 * (hi - lo), 0.4);
    ylim([lo - pad, hi + pad]);
end

set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
tighten_lr(gcf);
hold off;
