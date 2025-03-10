% model_num = 6;
% rate_num = 3;
% acc_snr_3 = zeros(rate_num,model_num);
% acc_snr_5 = zeros(rate_num,model_num);
% acc_snr_7 = zeros(rate_num,model_num);
% acc_snr_10 = zeros(rate_num,model_num);
% data_size = zeros(rate_num,model_num);

% Load data
load("acc_payload_data.mat");

% Define SNR values and corresponding accuracy matrices
snr_values = [3, 5, 7, 10];  
acc_matrices = {acc_snr_3, acc_snr_5, acc_snr_7, acc_snr_10}; 

% Define model labels and markers
model_labels = {'STD-12', 'BOX-12', 'STD-6', 'BOX-6', 'STD-3', 'BOX-3'};  
markers = {'o', 's', 'd', '^', 'v', 'p'}; % Different marker styles

% Define colors: STD (Blue shades), BOX (Green shades)
colors = [  % RGB values
    0, 0.2, 1;    % STD-12 (Deep Blue)
    0, 0.6, 0.2;  % BOX-12 (Deep Green)
    0, 0.4, 0.8;  % STD-6 (Medium Blue)
    0, 0.8, 0.4;  % BOX-6 (Medium Green)
    0.4, 0.7, 1;  % STD-3 (Light Blue)
    0.2, 1, 0.6   % BOX-3 (Light Green)
];

% Create figure with tight layout
figure;
tiledlayout(2,2,'TileSpacing','tight','Padding','tight'); 

% Loop through each SNR scenario
for i = 1:4
    nexttile;
    hold on;
    
    % Plot accuracy curves for each model
    for j = 1:6
        plot(data_size(:,j), acc_matrices{i}(:,j), '-','Marker', markers{j}, ...
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

% Create a shared legend and position it between the first and second row
lgd = legend(model_labels, 'FontSize', 12, 'FontName', 'Times New Roman', ...
             'NumColumns', 3, 'Location', 'northoutside'); % Place legend at the top

% Move the legend between the first and second row
lgd.Layout.Tile = 'north'; % Assign legend to the top space of the tiled layout
