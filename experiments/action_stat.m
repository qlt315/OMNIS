% Set up the colormap
cmap = lines(5);  % Default MATLAB color map (5 colors for 5 algorithms)
color_matrix = [];
for i = 1:5
    color_matrix = [color_matrix; cmap(i, :); cmap(i, :) * 0.7; cmap(i, :) * 0.4];  % Light and dark variations for each bar
end

% Plotting the results
figure('Position', [100, 100, 1600, 800]);  % Set figure size
for i = 1:6
    % Adjust position for more compact arrangement between rows
    if i <= 3
        % First row (SNR-related plots)
        ax = axes('Position', [0.1 + (mod(i-1, 3) * 0.3), 0.5, 0.25, 0.4]);  % Adjust position and size
    else
        % Second row (User-related plots)
        ax = axes('Position', [0.1 + (mod(i-4, 3) * 0.3), 0.05, 0.25, 0.4]);  % Adjust position and size
    end
    hold on;
    
    % Choose data based on SNR or Users
    if i <= 3
        data = squeeze(action_freq_snr(i, :, :));  % Use SNR-based data
        top3_data = top3_snr;
    else
        data = squeeze(action_freq_user_num(i-3, :, :));  % Use Users-based data
        top3_data = top3_user;
    end
    
    % Prepare a cell array to hold results for all algorithms
    top_3_results = cell(5, 1);
    
    % Plot each bar for 5 algorithms (5 bars per subplot)
    for alg = 1:5
        % Extract data for the algorithm and sort for top 3 actions
        selected_data = squeeze(data(alg, :));  % Get the action selection probabilities for the current algorithm
        if i <= 3
            top_3_values = selected_data(top3_snr{i, alg}(1:3));  % Top 3 actions based on the sorted order
        else
            top_3_values = selected_data(top3_user{i-3, alg}(1:3));  % Top 3 actions based on the sorted order
        end
        
        % Store the top 3 actions and their values for later output
        if i<=3
            top_3_results{alg} = [top_3_results{alg}; models(top3_snr{i, alg}(1:3)), num2cell(top_3_values)];  % Save actions and values
        else
            top_3_results{alg} = [top_3_results{alg}; models(top3_user{i-3, alg}(1:3)), num2cell(top_3_values)];  % Save actions and values
        end
        
        b = barh(alg, top_3_values, 'stacked', 'BarWidth', 0.6);  % Plot stacked bar
        % Assign colors to the 3 stacked parts
        set(b(1), 'FaceColor', color_matrix((alg-1)*3+1, :));
        set(b(2), 'FaceColor', color_matrix((alg-1)*3+2, :));
        set(b(3), 'FaceColor', color_matrix((alg-1)*3+3, :));
        
        % Annotate each bar with the action name, ensuring spaced-out placement
        for j = 1:3
            if i<= 3
                action_idx = top3_snr{i, alg}(j);  % Get index of the top 3 action
            else
                action_idx = top3_user{i-3, alg}(j);  % Get index of the top 3 action
            end
            action_name = models{action_idx};  % Get action name from the model list
            
            % Adjust the x-position to avoid overlap, and set y-position based on the algorithm
            text_position_x = top_3_values(j) + 0.05;  % Offset the x-position from the bars
            text_position_y = alg + 0.2 * (j-2);  % Adjust the y-position based on action number (spacing)
            
            % Display action name with adjusted positions
            text(text_position_x, text_position_y, action_name, 'FontSize', 14, 'FontName', 'Times New Roman', ...
                'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        end
    end
    
    % Set axis labels and titles
    xlabel('Action Pick Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
    yticks(1:5);
    set(gca, 'TickLabelInterpreter', 'tex');
    yticklabels({'OMNIS\newline  -UCB', 'CTO', 'OMNIS\newline   -TS', 'GDO', 'RSS'});

    yticklabels({'', '', '', '', ''});  % Algorithm names
    % Adjust X-axis range (max probability = 1)
    xlim([0, 1]);
    set(gca, 'FontSize', 14, 'FontName', 'Times New Roman');
    
    % Enable grid and other settings
    grid on;
    hold off;

    % Grid visibility improvement
    ax.GridColor = [0.2 0.2 0.2];  % Darker grid color
    ax.GridAlpha = 0.6;  % Increase grid opacity
    ax.Box = 'on';  % Display border
    
    % Output the top 3 actions and probabilities for each algorithm
    disp(['Results for subplot: ', titles{i}]);
    for alg = 1:5
        disp(['Algorithm: ', alg_names{alg}]);
        disp('Top 3 Actions and Probabilities:');
        disp(top_3_results{alg});
        disp('--------------------------------');
    end
end

% Apply tightfig to remove extra whitespace
tightfig;
