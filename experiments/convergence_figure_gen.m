% Load the .mat file containing the results
load('results/eval_convergence.mat');

% Extract the relevant data
metrics = {'reward', 'delay', 'accuracy', 'energy', 'is_vio', 'vio_degree'}; % Use actual metric names
metric_labels = {'Avg. Reward', 'Avg. Latency [s]', 'Avg. Acc. [%]', 'Avg. Energy [J]', 'Avg. Violation Prob.', ' Avg. Violation Excess'}; % Labels
algorithms = {'omnis', 'cto', 'dts', 'gdo', 'rss'};
algorithm_labels = {'OMNIS-UCB', 'CTO', 'OMNIS-TS', 'GDO', 'RSS'}; % 5 algorithms
user_num = 3; % Number of users
time_slot_num = 150; % Number of time slots

% Define colors and styles
colors = lines(length(algorithms)); % Assign colors
line_styles = {'--', ':', '-.'}; % Dashed line styles for individual users

% Create a figure with tighter spacing
figure('Position', [100, 100, 1000, 800]); % Adjust figure size

tiledlayout(2,3, 'Padding', 'compact', 'TileSpacing', 'compact'); % Use tiledlayout for tighter control

legend_entries = {}; % Initialize legend list

for metric_idx = 1:length(metrics)
    metric = metrics{metric_idx}; % Get metric name
    
    % Create subplot
    ax = nexttile; % Use tiled layout
    hold on;
    
    % OMNIS user-wise metrics
    omnis_ins_metric = eval(['omnis_ins_' metric]);


    % Plot other algorithms
    for alg_idx = 1:length(algorithms)
        alg_name = algorithms{alg_idx};
        alg_metric = eval([alg_name '_ins_' metric]);
        alg_metric_avg = mean(alg_metric, 1);
        plot(alg_metric_avg, 'LineWidth', 2.5, 'LineStyle', '-', 'Color', colors(alg_idx,:), 'DisplayName', algorithm_labels{alg_idx});
    end

    % Customize axes
    xlabel('Time Slot', 'FontSize', 14, 'FontName', 'Times New Roman');
    ylabel(metric_labels{metric_idx}, 'FontSize', 14, 'FontName', 'Times New Roman');
    
    % Improve grid visibility
    grid on;
    ax.GridColor = [0.2 0.2 0.2]; % Darker grid color
    ax.GridAlpha = 0.6; % Increase grid opacity
    ax.Box = 'on';
    
    % Set font properties
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');

    % Add tight margins
    set(gca, 'LooseInset', get(gca, 'TightInset'));
end

% Add a global legend at the bottom
lgd = legend(legend_entries, 'Orientation', 'horizontal', 'Location', 'southoutside', 'FontSize', 14, 'FontName', 'Times New Roman');
lgd.Box = 'off'; % Remove box around legend

% Ensure all text uses Times New Roman
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
