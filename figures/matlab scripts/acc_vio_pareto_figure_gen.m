% Acc–vio Pareto from figures/python figures/sweeps/acc_vio/acc_vio_pareto.mat
% Style aligned with conference OMNIS sweep scripts; does not save figures.
%
% Shared environment; OMNIS+ / GDO sweep feas_margin.

this_dir = fileparts(mfilename('fullpath'));
repo_root = fileparts(fileparts(this_dir));
addpath(this_dir);
mat_path = fullfile(repo_root, 'figures', 'python figures', 'sweeps', ...
    'acc_vio', 'acc_vio_pareto.mat');
data = load(mat_path);

algo_keys = {'causal', 'gdo'};
algo_labels = {'OMNIS+', 'GDO'};
colors = lines(numel(algo_keys));
markers = {'s', 'o'};
% OMNIS+: after sorting by vio, drop the suffix once Acc declines.
% Set to a positive int to force "first N points"; [] = auto (cut at first drop).
causal_keep_n = [];

figure('Position', [100, 100, 900, 220]);
hold on;

all_y = [];
all_ye = [];
for alg_idx = 1:numel(algo_keys)
    key = algo_keys{alg_idx};
    vio_f = sprintf('%s_vio_mean', key);
    acc_f = sprintf('%s_acc_mean', key);
    vio_s_f = sprintf('%s_vio_std', key);
    acc_s_f = sprintf('%s_acc_std', key);
    if ~isfield(data, vio_f) || ~isfield(data, acc_f)
        warning('Missing %s frontier fields; skip', key);
        continue;
    end
    x = double(data.(vio_f)(:)');
    y = double(data.(acc_f)(:)') * 100;  % Acc in %
    if isempty(x)
        continue;
    end
    [x, ord] = sort(x);
    y = y(ord);
    xe = zeros(size(x));
    ye = zeros(size(y));
    if isfield(data, vio_s_f)
        xe = double(data.(vio_s_f)(:)');
        xe = xe(ord);
    end
    if isfield(data, acc_s_f)
        ye = double(data.(acc_s_f)(:)') * 100;
        ye = ye(ord);
    end

    % OMNIS+: only the rising (non-decreasing Acc) prefix vs violation.
    if strcmp(key, 'causal')
        if isempty(causal_keep_n)
            keep = 1;
            for i = 2:numel(y)
                if y(i) + 1e-9 >= y(i - 1)
                    keep = i;
                else
                    break;
                end
            end
        else
            keep = min(causal_keep_n, numel(y));
        end
        x = x(1:keep); y = y(1:keep);
        xe = xe(1:keep); ye = ye(1:keep);
        fprintf('OMNIS+: keep %d / %d points (drop Acc-declining tail)\n', ...
            keep, numel(ord));
    end

    % Draw vertical and horizontal error bars separately so tiny Acc std
    % is not swallowed when x-err dominates (esp. GDO on a flat axes).
    errorbar(x, y, ye, 'vertical', '-', ...
        'LineWidth', 2.5, 'Color', colors(alg_idx, :), ...
        'Marker', markers{alg_idx}, 'MarkerSize', 9, ...
        'CapSize', 6, 'DisplayName', algo_labels{alg_idx});
    errorbar(x, y, xe, 'horizontal', 'LineStyle', 'none', ...
        'LineWidth', 2.5, 'Color', colors(alg_idx, :), ...
        'CapSize', 6, 'HandleVisibility', 'off');
    all_y = [all_y, y]; %#ok<AGROW>
    all_ye = [all_ye, ye]; %#ok<AGROW>
end

xlabel('Violation Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Avg. Acc. [%]', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
ax = gca;
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';
set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');

% Keep room so Acc error-bar caps are not clipped at the top/bottom.
if ~isempty(all_y)
    pad = max([all_ye(:); 0.15]);
    ylim([min(all_y) - 1.2 * pad, max(all_y) + 1.2 * pad]);
end

lgd = legend(algo_labels, 'Orientation', 'horizontal', ...
    'Location', 'best', 'FontSize', 14, ...
    'FontName', 'Times New Roman', 'NumColumns', numel(algo_labels));
lgd.Box = 'off';

text(mean(xlim), min(ylim) + 0.12 * (max(ylim) - min(ylim)), ...
    'Feasibility Margin Increases', ...
    'FontSize', 16, 'FontName', 'Times New Roman', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
    'Color', 'k');

set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
tighten_all(gcf);

