% Action-pick top-3 bars from action_pick_{snr,users}.mat
% Style aligned with conference action_stat.m; does not save figures.
% Algorithms: OMNIS+, C-OMNIS+, OMNIS, GDO, DQN, PPO, MAPPO (omits TS).
% Branch labels: B-3 / S-3 centered inside each stack segment.
% Y-tick labels only on the left column; flat figure, FontSize 14.

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(this_dir);
snr_mat = load(fullfile(repo_root, 'figures', 'python figures', 'sweeps', 'action_pick', 'action_pick_snr.mat'));
usr_mat = load(fullfile(repo_root, 'figures', 'python figures', 'sweeps', 'action_pick', 'action_pick_users.mat'));

% Display order for models (matches mat field suffixes)
model_keys = {'Box3', 'Box6', 'Box12', 'Standard3', 'Standard6', 'Standard12'};
model_labels = {'Box-3', 'Box-6', 'Box-12', 'Standard-3', 'Standard-6', 'Standard-12'};
model_short = {'B-3', 'B-6', 'B-12', 'S-3', 'S-6', 'S-12'};

algo_keys = {'causal', 'cto', 'ucb', 'gdo', 'dqn', 'ppo', 'mappo'};
algo_ylabels = {'OMNIS+', 'C-OMNIS+', 'OMNIS', 'GDO', 'DQN', 'PPO', 'MAPPO'};

n_algo = numel(algo_keys);
n_model = numel(model_keys);

% Per-algo family color with 3 shades for stacked top-3
cmap = algo_family_colors(algo_keys);
color_matrix = zeros(n_algo * 3, 3);
for i = 1:n_algo
    color_matrix((i - 1) * 3 + 1, :) = cmap(i, :);
    color_matrix((i - 1) * 3 + 2, :) = local_lighten(cmap(i, :), 0.25);
    color_matrix((i - 1) * 3 + 3, :) = local_lighten(cmap(i, :), 0.45);
end

snr_settings = cellstr(string(snr_mat.settings(:)));
usr_settings = cellstr(string(usr_mat.settings(:)));
assert(numel(snr_settings) >= 3 && numel(usr_settings) >= 3, ...
    'Need at least 3 settings per axis');

% Wide + short (flat); left margin reserved for y-tick labels only
figure('Position', [80, 80, 1500, 480]);

min_label_w = 0.06;
fs = 14;

% Left column needs room for "C-OMNIS+"; mid/right have no y-labels → tight gaps
n_col = 3;
n_row = 2;
left_margin = 0.105;
right_margin = 0.015;
bottom_margin = 0.12;
top_margin = 0.05;
col_gap = 0.018;
row_gap = 0.08;
col_w = (1 - left_margin - right_margin - (n_col - 1) * col_gap) / n_col;
row_h = (1 - bottom_margin - top_margin - (n_row - 1) * row_gap) / n_row;

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
    ax = axes('Position', ...
        [left_margin + (col - 1) * (col_w + col_gap), ...
         bottom_margin + (n_row - row) * (row_h + row_gap), ...
         col_w, row_h]);
    hold on;

    for alg = 1:n_algo
        [~, sorted_idx] = sort(data(alg, :), 'descend');
        top3 = sorted_idx(1:min(3, n_model));
        top_vals = data(alg, top3);
        b = barh(alg, top_vals, 'stacked', 'BarWidth', 0.65);
        x0 = 0;
        for j = 1:numel(top_vals)
            set(b(j), 'FaceColor', color_matrix((alg - 1) * 3 + j, :));
            w = top_vals(j);
            if w >= min_label_w
                text(x0 + w / 2, alg, model_short{top3(j)}, ...
                    'FontSize', fs, 'FontName', 'Times New Roman', ...
                    'Color', 'w', 'FontWeight', 'bold', ...
                    'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', ...
                    'Clipping', 'on');
            end
            x0 = x0 + w;
        end
    end

    title(condition_label, 'FontSize', fs, 'FontName', 'Times New Roman');
    if row == n_row
        xlabel('Action Pick Probability', 'FontSize', fs, 'FontName', 'Times New Roman');
    end
    yticks(1:n_algo);
    if col == 1
        yticklabels(algo_ylabels);
    else
        yticklabels({});
    end
    xlim([0, 1.02]);
    ylim([0.55, n_algo + 0.45]);
    set(ax, 'FontSize', fs, 'FontName', 'Times New Roman', 'YDir', 'reverse');
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
set(findall(gcf, '-property', 'FontSize'), 'FontSize', fs);

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

function c = local_lighten(base, t)
    c = min(max(base + (1 - base) * t, 0), 1);
end
