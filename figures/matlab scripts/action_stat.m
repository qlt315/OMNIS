% Action-pick top-3 bars from action_pick_{snr,users}.mat
% Style aligned with conference action_stat.m; does not save figures.
% Algorithms: OMNIS+, OMNIS, GDO, PPO, MAPPO, CTO (no DQN/TS).

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(this_dir);
snr_mat = load(fullfile(repo_root, 'figures', 'python figures', 'sweeps', 'action_pick', 'action_pick_snr.mat'));
usr_mat = load(fullfile(repo_root, 'figures', 'python figures', 'sweeps', 'action_pick', 'action_pick_users.mat'));

% Display order for models (matches mat field suffixes)
model_keys = {'Box3', 'Box6', 'Box12', 'Standard3', 'Standard6', 'Standard12'};
model_labels = {'Box-3', 'Box-6', 'Box-12', 'Standard-3', 'Standard-6', 'Standard-12'};

algo_keys = {'causal', 'ucb', 'gdo', 'ppo', 'mappo', 'cto'};
algo_ylabels = {'OMNIS+', 'OMNIS', 'GDO', 'PPO', 'MAPPO', 'CTO'};

n_algo = numel(algo_keys);
n_model = numel(model_keys);

% Per-algo color with 3 shades for stacked top-3
cmap = lines(n_algo);
color_matrix = zeros(n_algo * 3, 3);
for i = 1:n_algo
    color_matrix((i - 1) * 3 + 1, :) = cmap(i, :);
    color_matrix((i - 1) * 3 + 2, :) = cmap(i, :) * 0.75;
    color_matrix((i - 1) * 3 + 3, :) = cmap(i, :) * 0.5;
end

snr_settings = cellstr(string(snr_mat.settings(:)));
usr_settings = cellstr(string(usr_mat.settings(:)));
assert(numel(snr_settings) >= 3 && numel(usr_settings) >= 3, ...
    'Need at least 3 settings per axis');

figure('Position', [80, 80, 1600, 820]);

for i = 1:6
    if i <= 3
        cond_idx = i;
        data = local_pick_matrix(snr_mat, algo_keys, model_keys, cond_idx);
        condition_label = snr_settings{cond_idx};
    else
        cond_idx = i - 3;
        data = local_pick_matrix(usr_mat, algo_keys, model_keys, cond_idx);
        condition_label = usr_settings{cond_idx};
    end

    row = floor((i - 1) / 3) + 1;
    col = mod(i - 1, 3) + 1;
    % Slightly wider column stride so y-tick labels do not collide with the
    % neighboring axes (gap ≈ 0.055 of figure width).
    ax = subplot(2, 3, i, 'Position', ...
        [0.035 + (col - 1) * 0.335, 0.53 - (row - 1) * 0.41, 0.285, 0.38]);
    hold on;

    for alg = 1:n_algo
        [~, sorted_idx] = sort(data(alg, :), 'descend');
        top3 = sorted_idx(1:min(3, n_model));
        top_vals = data(alg, top3);
        b = barh(alg, top_vals, 'stacked', 'BarWidth', 0.6);
        for j = 1:numel(top_vals)
            set(b(j), 'FaceColor', color_matrix((alg - 1) * 3 + j, :));
            text(sum(top_vals(1:j)) + 0.01, alg + 0.18 * (j - 2), ...
                model_labels{top3(j)}, ...
                'FontSize', 13, 'FontName', 'Times New Roman', ...
                'Color', 'k', 'FontWeight', 'bold', ...
                'HorizontalAlignment', 'left');
        end
    end

    if row == 2
        xlabel('Action Pick Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
    end
    yticks(1:n_algo);
    yticklabels(algo_ylabels);
    xlim([0, 1.15]);
    % Hug the bars: default ylim [0.5, n+0.5] leaves empty bands above OMNIS+ / below CTO.
    ylim([0.65, n_algo + 0.35]);
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman', 'YDir', 'reverse');
    grid on;
    ax.GridColor = [0.2 0.2 0.2];
    ax.GridAlpha = 0.6;
    ax.Box = 'on';
    hold off;

    fprintf('Results for %s\n', condition_label);
    for alg = 1:n_algo
        [~, sorted_idx] = sort(data(alg, :), 'descend');
        top3 = sorted_idx(1:3);
        fprintf('  %s: ', algo_ylabels{alg});
        for j = 1:3
            fprintf('%s=%.3f ', model_labels{top3(j)}, data(alg, top3(j)));
        end
        fprintf('\n');
    end
end

set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
tighten_lr(gcf);

function P = local_pick_matrix(mat, algo_keys, model_keys, setting_idx)
%LOCAL_PICK_MATRIX Build [n_algo x n_model] pick probs for one setting.
    n_a = numel(algo_keys);
    n_m = numel(model_keys);
    P = zeros(n_a, n_m);
    for a = 1:n_a
        for m = 1:n_m
            fname = sprintf('pick_%s_%s', algo_keys{a}, model_keys{m});
            if isfield(mat, fname)
                v = double(mat.(fname)(:));
                P(a, m) = v(min(setting_idx, numel(v)));
            end
        end
    end
end
