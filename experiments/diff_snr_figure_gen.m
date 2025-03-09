% Define marker styles for each algorithm
marker_styles = {'-o', '-s', '-d', '-^', '-v'}; % Different markers for each algorithm

% Load the .mat file containing performance metrics at different SNR levels
data = load('results/eval_diff_snr.mat');

% Define SNR values and algorithm names
snr_values = [0, 2, 4, 6, 8, 10]; % SNR levels
algorithms = {'OMNIS', 'CTO', 'DTS', 'GDO', 'RSS'}; % 5 algorithms
metrics = {'reward', 'latency', 'energy', 'accuracy', 'vio_prob', 'vio_sum'}; % 6 performance metrics
metric_labels = {'Aver. Reward of MDs', 'Aver. Latency [s]', 'Aver. Energy [J]', 'Aver. Acc. [%]', 'Aver. Vio. Prob.', 'Aver. Vio. Sum'}; % Y-axis labels

% Define color scheme for different algorithms
colors = lines(length(algorithms)); % Automatically assign different colors

% Create a figure with a 2x3 subplot layout and reduce the size of the figure
figure('Position', [100, 100, 1100, 500]); % Adjust figure size

% Tight layout to remove empty space around the figure
set(gcf, 'PaperPositionMode', 'auto'); 

for metric_idx = 1:length(metrics)
    metric_name = metrics{metric_idx}; % Get current metric name
    matrix_name = [metric_name '_diff_snr']; % Construct the matrix name (e.g., 'reward_diff_snr')
    
    if isfield(data, matrix_name)
        metric_data = data.(matrix_name); % Extract corresponding data matrix (5x6)
    else
        warning('Matrix "%s" not found in eval_diff_snr.mat', matrix_name);
        continue;
    end

    % Create a subplot (2 rows, 3 columns)
    ax = subplot(2, 3, metric_idx);
    hold on;

    % Plot data for each algorithm with different markers
    for alg_idx = 1:length(algorithms)
        plot(snr_values, metric_data(alg_idx, :), marker_styles{alg_idx}, 'LineWidth', 3, ...
             'Color', colors(alg_idx, :), 'DisplayName', algorithms{alg_idx}, 'MarkerSize', 10);
    end

    % Set axis labels
    xlabel('SNR [dB]', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel(metric_labels{metric_idx}, 'FontSize', 12, 'FontName', 'Times New Roman');

    
    % Set x-axis ticks to show user counts 1 to 8
    xticks(0:2:10);
    xlim([0, 10]);

    % Enable grid and adjust grid color for better visibility
    grid on;
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman', 'GridColor', [0.3, 0.3, 0.3]);

    % Improve grid visibility
    grid on;
    ax.GridColor = [0.2 0.2 0.2]; % Darker grid color
    ax.GridAlpha = 0.6; % Increase grid opacity
    ax.Box = 'on';

end

% Create a shared legend below all subplots
hL = legend(algorithms, 'Orientation', 'horizontal', 'FontSize', 14, 'FontName', 'Times New Roman', 'NumColumns', 5);
newPosition = [0.2, 0.01, 0.6, 0.05]; % Adjust legend position to be more compact
newUnits = 'normalized';
set(hL, 'Position', newPosition, 'Units', newUnits, 'Box', 'off', 'Color', 'none'); % Make legend transparent

% Tighten the overall layout to minimize empty space around the entire figure
set(gcf, 'PaperPositionMode', 'auto');
tightfig; % A function to remove extra white space from the figure

