% Acc–vio Pareto from figures/python figures/sweeps/acc_vio/acc_vio_pareto.mat
% Style aligned with conference OMNIS sweep scripts; does not save figures.
%
% Shared environment; OMNIS+ / GDO sweep feas_margin (annotated on markers).

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

figure('Position', [100, 100, 720, 560]);
hold on;

for alg_idx = 1:numel(algo_keys)
    key = algo_keys{alg_idx};
    vio_f = sprintf('%s_vio_mean', key);
    acc_f = sprintf('%s_acc_mean', key);
    vio_s_f = sprintf('%s_vio_std', key);
    acc_s_f = sprintf('%s_acc_std', key);
    knob_f = sprintf('%s_knob', key);
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
    errorbar(x, y, ye, ye, xe, xe, '-', ...
        'LineWidth', 2.5, 'Color', colors(alg_idx, :), ...
        'Marker', markers{alg_idx}, 'MarkerSize', 9, ...
        'CapSize', 4, 'DisplayName', algo_labels{alg_idx});

    if isfield(data, knob_f)
        kn = double(data.(knob_f)(:)');
        kn = kn(ord);
        for i = 1:numel(x)
            text(x(i), y(i), sprintf('  %.2f', kn(i)), ...
                'Color', colors(alg_idx, :), ...
                'FontSize', 12, 'FontName', 'Times New Roman', ...
                'VerticalAlignment', 'bottom');
        end
    end
end

xlabel('Violation Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Avg. Acc. [%]', 'FontSize', 14, 'FontName', 'Times New Roman');
grid on;
ax = gca;
ax.GridColor = [0.2 0.2 0.2];
ax.GridAlpha = 0.6;
ax.Box = 'on';
set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');

lgd = legend(algo_labels, 'Orientation', 'horizontal', ...
    'Location', 'southoutside', 'FontSize', 12, ...
    'FontName', 'Times New Roman', 'NumColumns', numel(algo_labels));
lgd.Box = 'off';
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
tighten_lr(gcf);
