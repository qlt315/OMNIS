model_num = 6;
snr_num = 5;
rate_num = 3;
% acc_rate_1 = zeros(model_num, snr_num); % rate = 1
% acc_rate_2 = zeros(model_num, snr_num); % rate = 2/3
% acc_rate_3 = zeros(model_num, snr_num);  % rate = 5/6
load("acc_snr_data.mat")


% Define SNR values and corresponding accuracy matrices
snr_values = [0, 3, 5, 7, 10];  
rate_matrices = {acc_rate_1, acc_rate_2, acc_rate_3}; 

% Define model labels and markers
model_labels = {'STD-12', 'STD-6', 'STD-3', 'ENT-12', 'ENT-6', 'ENT-3'};  % Added Entropy model
markers = {'o', 'd', 'v', 'x', '*', 'h'}; % Expanded to 9 markers


% Define colors: STD (Blue shades), BOX (Green shades), and Entropy (Red shades)
colors = [  % RGB values
    0, 0.2, 1;    % STD-12 (Deep Blue)
    0, 0.4, 0.8;  % STD-6 (Medium Blue)
    0.4, 0.7, 1;  % STD-3 (Light Blue)
    0.8, 0, 0;    % ENT-3 (Deep Red)
    1, 0.4, 0.4;  % ENT-6 (Medium Red)
    1, 0.7, 0.7;  % ENT-12 (Light Red)
];


% Create figure with tight layout
figure;
tiledlayout(1, 3, 'TileSpacing', 'tight', 'Padding', 'tight'); 


% Loop through each rate scenario
for i = 1:rate_num
    nexttile;
    hold on;
    
    % Plot accuracy curves for each model (including Entropy)
    for j = 1:model_num
        plot(snr_values, rate_matrices{i}(j,:), '-','Marker', markers{j}, ...
             'Color', colors(j,:), 'LineWidth', 2, 'MarkerSize', 8, ...
             'DisplayName', model_labels{j});
    end
    
    % Set axis labels
    xlabel('SNR (dB)', 'FontSize', 14, 'FontName', 'Times New Roman');
    ylabel('Accuracy', 'FontSize', 14, 'FontName', 'Times New Roman');
    
    % Customize grid style
    grid on;
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman', ...
             'GridColor', [0.3, 0.3, 0.3], 'GridAlpha', 0.6, 'Box', 'on');
    
    hold off;
end

% Create a shared legend and position it in a single row
lgd = legend(model_labels, 'FontSize', 12, 'FontName', 'Times New Roman', ...
             'NumColumns', length(model_labels), 'Location', 'northoutside'); % Single-row legend

% Move the legend between the first and second row
lgd.Layout.Tile = 'north'; % Assign legend to the top space of the tiled layout