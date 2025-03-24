% Load data
load("results/action_freq_snr.mat");  % Shape: 3×num_seeds×5×6
load("results/action_freq_user_num.mat");  % Shape: 3×num_seeds×5×6

% Define models in MATLAB struct format
models = struct('name', {}, 'quant_method', {}, 'quant_channel', {});
quant_methods = {'Box-', 'Box-', 'Box-', 'Standard-', 'Standard-', 'Standard-'};
quant_channels = [3, 6, 12, 3, 6, 12];

for i = 1:length(quant_methods)
    models(i).name = sprintf('%s%d', quant_methods{i}, quant_channels(i));
    models(i).quant_method = quant_methods{i};
    models(i).quant_channel = quant_channels(i);
end

% Set up colormap
cmap = lines(5);  % Default MATLAB colormap (5 colors for 5 algorithms)
color_matrix = [];
for i = 1:5
    color_matrix = [color_matrix; cmap(i, :); cmap(i, :) * 0.7; cmap(i, :) * 0.4];  % Light and dark variations
end

% SNR and MD labels
snr_values = [2, 4, 6];
md_values = [2, 4, 6];

% Plotting the results (6 subplots: 3 for SNR, 3 for MD count)
figure('Position', [100, 100, 1600, 800]);  % Set figure size

% Adjust the subplot position to reduce space between rows and keep horizontal distance
for i = 1:6
    % Determine whether we are plotting SNR or MD
    if i <= 3
        condition_idx = i;  % SNR index (1,2,3)
        data = squeeze(action_freq_snr(condition_idx, :, :));  % Shape: 5×6
        condition_label = sprintf('SNR = %d', snr_values(i));
    else
        condition_idx = i - 3;  % MD index (1,2,3)
        data = squeeze(action_freq_user_num(condition_idx, :, :));  % Shape: 5×6
        condition_label = sprintf('MD Count = %d', md_values(condition_idx));
    end
    
    % Find top-3 actions for each algorithm
    top3_data = cell(1, 5);
    for alg = 1:5
        [~, sorted_indices] = sort(data(alg, :), 'descend');
        top3_data{alg} = sorted_indices(1:3);
    end
    
    % Create subplot with adjusted position (tight position between rows and proper horizontal spacing)
    row = floor((i-1) / 3) + 1; % Calculate the row number
    col = mod(i-1, 3) + 1; % Calculate the column number
    ax = subplot(2, 3, i, 'Position', [0.02 + (col-1)*0.33, 0.55 - (row-1)*0.44, 0.28, 0.4]);

    hold on;
    
    % Prepare a cell array to hold top-3 results
    top_3_results = cell(5, 1);
    
    % Plot each algorithm (5 bars per subplot)
    for alg = 1:5
        top_3_indices = top3_data{alg};
        top_3_values = data(alg, top_3_indices);
        
        % Store top 3 actions and probabilities
        top_3_names = reshape({models(top_3_indices).name}, [], 1);
        top_3_results{alg} = [top_3_names; num2cell(top_3_values')];
        
        % Plot stacked bars
        b = barh(alg, top_3_values, 'stacked', 'BarWidth', 0.6);
        for j = 1:3
            set(b(j), 'FaceColor', color_matrix((alg-1)*3+j, :));
        end
        
        % Annotate each bar
        for j = 1:3
            action_name = models(top_3_indices(j)).name;
            text(top_3_values(j) + 0.05, alg + 0.2 * (j-2), action_name, ...
                'FontSize', 14, 'FontName', 'Times New Roman', 'HorizontalAlignment', 'left');
        end
    end
    
    % Set axis labels
    xlabel('Action Pick Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
    yticks(1:5);
    yticklabels({'OMNIS\newline  -UCB', 'CTO', 'OMNIS\newline   -TS', 'GDO', 'RSS'});
    xlim([0, 1]);
    
    % Set grid
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
    grid on;
    hold off;

    % Improve grid visibility
    ax.GridColor = [0.2 0.2 0.2];
    ax.GridAlpha = 0.6;
    ax.Box = 'on';

    % Output top-3 results
    disp(['Results for ', condition_label]);
    for alg = 1:5
        disp(['Algorithm: ', num2str(alg)]);
        disp('Top 3 Actions and Probabilities:');
        disp(top_3_results{alg});
        disp('--------------------------------');
    end
end

% Apply tight figure layout
tightfig;
