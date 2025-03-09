% Load the MAT files containing action selection probabilities
snr_data = load('results/action_freq_snr.mat');
user_data = load('results/action_freq_user_num.mat');

% Extract the action frequency matrices (3 SNR levels / user counts, 5 algorithms, 6 actions)
action_freq_snr = snr_data.action_freq_snr; % Size: 3x5x6
action_freq_users = user_data.action_freq_user_num; % Corrected matrix name

% Define SNR values and user count values (reversed order for top-to-bottom plotting)
snr_values = [6, 4, 2];  % Top to bottom order
user_counts = [6, 4, 2];

algorithms = {'OMNIS', 'CTO', 'DTS', 'GDO', 'RSS'}; % 5 algorithms
num_algorithms = length(algorithms);
num_conditions = length(snr_values); % 3 conditions (SNR or user count)

% Define model names corresponding to action indices
model_names = {'Box3', 'Box6', 'Box12', 'Standard3', 'Standard6', 'Standard12'};

% Find the most selected action and its probability for each algorithm in different conditions
[max_prob_snr, max_action_snr] = max(action_freq_snr, [], 3); % Highest probability and action index for SNR
[max_prob_user, max_action_user] = max(action_freq_users, [], 3); % Highest probability and action index for user count

% Reverse the order for correct display from top to bottom
max_prob_snr = flipud(max_prob_snr);
max_action_snr = flipud(max_action_snr);
max_prob_user = flipud(max_prob_user);
max_action_user = flipud(max_action_user);

% Create a figure with two subplots
figure('Position', [100, 100, 1400, 600]); % Set figure size

% Define color scheme
colors = lines(num_algorithms);

%% ---- Plot for SNR-based action selection (left subplot) ---- %%
ax1 = subplot(1, 2, 1);
hold on;
b = barh(1:num_algorithms, max_prob_snr', 'grouped'); % Plot horizontal bars
set(b, 'FaceColor', 'flat', 'EdgeColor', 'none'); % Remove edge color

% Add textures (hatch patterns) to each bar
for i = 1:num_algorithms
    for j = 1:num_conditions
        h = b(j).FaceColor; % Get the color for the bar
        hatchfill(b(j), 'single', 'HatchAngle', 45, 'HatchDensity', 10); % Apply hatch fill
    end
end

xlabel('Selection Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Algorithm', 'FontSize', 14, 'FontName', 'Times New Roman');
yticks(1:num_algorithms);
yticklabels(algorithms);
xlim([0, 1]); % Probability values range from 0 to 1

% Improve grid visibility
grid on;
ax1.GridColor = [0.2, 0.2, 0.2]; % Darker grid color
ax1.GridAlpha = 0.6; % Increase grid opacity
ax1.Box = 'on';

legend(arrayfun(@(x) sprintf('SNR=%d', x), snr_values, 'UniformOutput', false), ...
    'Location', 'northeast', 'FontSize', 12, 'FontName', 'Times New Roman');

% Label bars with the most selected action index (converted to model name)
for alg = 1:num_algorithms
    for cond = 1:num_conditions
        model_index = max_action_snr(cond, alg); % Get the action index
        text(max_prob_snr(cond, alg), alg - 0.3 + 0.3 * cond, model_names{model_index}, ... % Convert to model name
            'HorizontalAlignment', 'left', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    end
end
hold off;

%% ---- Plot for User Count-based action selection (right subplot) ---- %%
ax2 = subplot(1, 2, 2);
hold on;
b2 = barh(1:num_algorithms, max_prob_user', 'grouped'); % Plot horizontal bars
set(b2, 'FaceColor', 'flat', 'EdgeColor', 'none'); % Remove edge color

% Add textures (hatch patterns) to each bar
for i = 1:num_algorithms
    for j = 1:num_conditions
        h = b2(j).FaceColor; % Get the color for the bar
        hatchfill(b2(j), 'single', 'HatchAngle', 45, 'HatchDensity', 10); % Apply hatch fill
    end
end

xlabel('Selection Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
ylabel('Algorithm', 'FontSize', 14, 'FontName', 'Times New Roman');
yticks(1:num_algorithms);
yticklabels(algorithms);
xlim([0, 1]); % Probability values range from 0 to 1

% Improve grid visibility
grid on;
ax2.GridColor = [0.2, 0.2, 0.2]; % Darker grid color
ax2.GridAlpha = 0.6; % Increase grid opacity
ax2.Box = 'on';

legend(arrayfun(@(x) sprintf('Users=%d', x), user_counts, 'UniformOutput', false), ...
    'Location', 'northeast', 'FontSize', 12, 'FontName', 'Times New Roman');

% Label bars with the most selected action index (converted to model name)
for alg = 1:num_algorithms
    for cond = 1:num_conditions
        model_index = max_action_user(cond, alg); % Get the action index
        text(max_prob_user(cond, alg), alg - 0.3 + 0.3 * cond, model_names{model_index}, ... % Convert to model name
            'HorizontalAlignment', 'left', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Times New Roman');
    end
end
hold off;
