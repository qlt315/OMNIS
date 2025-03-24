load("acc_payload_data.mat");

% Define SNR values to be plotted (only 3dB and 10dB)
snr_values = [3, 10];  
acc_matrices = {acc_snr_3, acc_snr_10}; % Corresponding accuracy matrices

% Define model labels and markers (only STD-3, STD-6, and STD-12)
model_labels = {'STD-12', 'STD-6', 'STD-3'};
markers = {'o', 'd', 'v'}; % Markers for STD models

% Define colors for STD models (Blue shades)
colors = [  
    0, 0.2, 1;    % STD-12 (Deep Blue)
    0, 0.4, 0.8;  % STD-6 (Medium Blue)
    0.4, 0.7, 1;  % STD-3 (Light Blue)
];

% Create figure with a tiled layout
figure;
tiledlayout(1, 2, 'TileSpacing', 'tight', 'Padding', 'tight'); % 1 row, 2 columns

% Loop through each selected SNR scenario
for i = 1:2
    nexttile;
    hold on;
    
    % Plot accuracy curves for STD-3, STD-6, and STD-12
    for j = 1:3
        plot(data_size(:, j), acc_matrices{i}(:, j), '-', 'Marker', markers{j}, ...
             'Color', colors(j,:), 'LineWidth', 2, 'MarkerSize', 8, ...
             'DisplayName', model_labels{j});
    end
    
    % Set axis labels
    xlabel('Payload Size (KB)', 'FontSize', 14, 'FontName', 'Times New Roman');
    ylabel('Accuracy', 'FontSize', 14, 'FontName', 'Times New Roman');

    
    % Customize grid style
    grid on;
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman', ...
             'GridColor', [0.3, 0.3, 0.3], 'GridAlpha', 0.6, 'Box', 'on');
    
    hold off;
end

% Create a shared legend and position it at the top
lgd = legend(model_labels, 'FontSize', 12, 'FontName', 'Times New Roman', ...
             'NumColumns', 3, 'Location', 'northoutside');
lgd.Layout.Tile = 'north'; % Assign legend to the top space of the tiled layout
