% Stacked runtime bar from figures/convergence/plot_data.mat
% Stack: selection + update + interaction + BCD (ms/slot). Does not save figures.
% Plots OMNIS (better of UCB/TS); omits DQN and OMNIS-TS.

clear; close all; clc;

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
S = load(fullfile(repo_root, 'figures', 'convergence', 'plot_data.mat'));

algo_keys = {'causal', 'ucb', 'gdo', 'rss', 'ppo', 'mappo', 'cto'};
algo_labels = {'OMNIS+', 'OMNIS', 'GDO', 'RSS', 'PPO', 'MAPPO', 'CTO'};

% Map keys -> indices in algo_names
all_names = cellstr(string(S.algo_names(:)));
idx = zeros(1, numel(algo_keys));
for i = 1:numel(algo_keys)
    k = find(strcmp(all_names, algo_keys{i}), 1);
    if isempty(k)
        error('Algorithm "%s" not found in plot_data.mat', algo_keys{i});
    end
    idx(i) = k;
end

% Prefer summary fields (same order as algo_names); fall back to flat vectors
sm = S.summary;
select_ms = double(sm.select_ms(idx));
update_ms = double(sm.update_ms(idx));
comm_ms = double(sm.comm_ms(idx));
bcd_ms = double(sm.bcd_ms(idx));

stack = [select_ms(:), update_ms(:), comm_ms(:), bcd_ms(:)];  % [n_algo x 4]
stack_labels = {'selection', 'update', 'interaction', 'BCD'};
stack_colors = [0.298, 0.471, 0.659;  % #4c78a8
                0.620, 0.792, 0.914;  % #9ecae9
                0.329, 0.635, 0.294;  % #54a24b
                0.961, 0.522, 0.094]; % #f58518

figure('Position', [120, 120, 900, 480]);
ax = gca;
hold on;
b = bar(stack, 'stacked', 'BarWidth', 0.7);
for j = 1:numel(stack_labels)
    b(j).FaceColor = stack_colors(j, :);
    b(j).DisplayName = stack_labels{j};
end

set(gca, 'XTick', 1:numel(algo_labels), 'XTickLabel', algo_labels, ...
    'FontSize', 13, 'FontName', 'Times New Roman');
xtickangle(18);
ylabel('Runtime [ms/slot]', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.5;
ax.Box = 'on';
lgd = legend('Location', 'northwest', 'FontSize', 12, 'FontName', 'Times New Roman');
lgd.Box = 'off';
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
hold off;

% Print totals for quick check
tot = sum(stack, 2);
fprintf('Runtime totals (ms/slot):\n');
for i = 1:numel(algo_labels)
    fprintf('  %-14s  select=%.2f  update=%.2f  comm=%.2f  bcd=%.2f  total=%.2f\n', ...
        algo_labels{i}, select_ms(i), update_ms(i), comm_ms(i), bcd_ms(i), tot(i));
end
