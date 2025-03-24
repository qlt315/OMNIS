% Load the .mat file containing the results
load('results/eval_convergence.mat');

% Extract the relevant data
metrics = {'reward', 'delay', 'accuracy', 'energy', 'is_vio', 'vio_degree'}; % Use actual metric names
metric_labels = {'Avg. Reward', 'Avg. Latency [s]', 'Avg. Acc. [%]', 'Avg. Energy [J]', 'Avg. Violation Prob.', ' Avg. Violation Excess'}; % Labels for the metrics
algorithms = {'omnis', 'cto', 'dts', 'gdo', 'rss'}; % List of algorithms
algorithm_labels = {'OMNIS-UCB', 'CTO', 'OMNIS-TS', 'GDO', 'RSS'}; % Labels for the algorithms



% Define colors for each algorithm
colors = lines(length(algorithms)); % Generate distinct colors for each algorithm
line_styles = {'--', ':', '-.'}; % Dashed line styles for individual users (not used in this version)

% Create a figure with tighter spacing
figure('Position', [100, 100, 1000, 800]); % Adjust figure size for better presentation
tiledlayout(2,3, 'Padding', 'compact', 'TileSpacing', 'compact'); % Use tiledlayout for organized plots

legend_entries = {}; % Initialize list for legend entries

for metric_idx = 1:length(metrics)
    metric = metrics{metric_idx}; % Get metric name for current subplot
    
    % Create a subplot for each metric
    ax = nexttile; % Use nexttile for organized subplots
    hold on;
    

    % Plot metrics for all algorithms
    for alg_idx = 1:length(algorithms)
        alg_name = algorithms{alg_idx}; % Algorithm name
        alg_metric = eval([alg_name '_ins_' metric]); % Extract current algorithm's metric data
        alg_metric_avg = mean(alg_metric, 1); % Compute average across users
        
        % Apply smoothing to the average metric data (choose one method)
        % alg_metric_smooth = smoothdata(alg_metric_avg, 'movmean', 5); % Apply moving average (window size = 5)
        alg_metric_smooth = sgolayfilt(alg_metric_avg, 3, 49); % Apply Savitzky-Golay filter (3rd order, window size = 9)
        
        % Plot the smoothed metric data
        plot(alg_metric_avg, 'LineWidth', 2.5, 'LineStyle', '-', 'Color', colors(alg_idx,:), 'DisplayName', algorithm_labels{alg_idx});
    end

    % Customize axes for better readability
    xlabel('Time Slot', 'FontSize', 14, 'FontName', 'Times New Roman'); % X-axis label
    ylabel(metric_labels{metric_idx}, 'FontSize', 14, 'FontName', 'Times New Roman'); % Y-axis label
    
    % Improve grid visibility for better presentation
    grid on;
    ax.GridColor = [0.2 0.2 0.2]; % Darker grid color
    ax.GridAlpha = 0.6; % Increase grid opacity for better visibility
    ax.Box = 'on'; % Enable box around the plot for better clarity
    
    % Set font properties for axis labels and ticks
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');

    % Adjust tight margins for better layout
    set(gca, 'LooseInset', get(gca, 'TightInset'));
end

% Add a global legend at the bottom of the plot
lgd = legend(legend_entries, 'Orientation', 'horizontal', 'Location', 'southoutside', 'FontSize', 14, 'FontName', 'Times New Roman');
lgd.Box = 'off'; % Remove box around the legend for cleaner appearance

% Ensure that all text uses Times New Roman font
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
