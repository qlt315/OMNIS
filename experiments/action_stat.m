% Set up the colormap
cmap = lines(5);  % Default MATLAB color map (5 colors for 5 algorithms)
color_matrix = [];
for i = 1:5
    % Create a range of variations for each algorithm, from light to dark shades
    color_matrix = [color_matrix; cmap(i, :); cmap(i, :) * 0.85; cmap(i, :) * 0.6];  % Light, medium, and dark variations
end

% Plotting the results
figure('Position', [100, 100, 1600, 800]);  % Set figure size
for i = 1:6
    % Adjust position for tighter spacing between rows
    if i <= 3
        ax = subplot(2, 3, i, 'Position', [0.1 + (i-1)*0.3, 0.55, 0.25, 0.4]);  % Upper row (tighter spacing)
    else
        ax = subplot(2, 3, i, 'Position', [0.1 + (i-4)*0.3, 0.1, 0.25, 0.4]);  % Lower row (tighter spacing)
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
    
    % Plot each bar for 5 algorithms (5 bars per subplot)
    for alg = 1:5
        % Extract data for the algorithm and sort for top 3 actions
        selected_data = squeeze(data(alg, :));  % Get the action selection probabilities for the current algorithm
        if i <= 3
            top_3_values = selected_data(top3_snr{i, alg}(1:3));  % Top 3 actions based on the sorted order
        else
            top_3_values = selected_data(top3_user{i-3, alg}(1:3));  % Top 3 actions based on the sorted order
        end
        
        b = barh(alg, top_3_values, 'stacked', 'BarWidth', 0.6);  % Plot stacked bar
        % Assign colors to the 3 stacked parts
        set(b(1), 'FaceColor', color_matrix((alg-1)*3+1, :));
        set(b(2), 'FaceColor', color_matrix((alg-1)*3+2, :));
        set(b(3), 'FaceColor', color_matrix((alg-1)*3+3, :));
        
        % Annotate each bar with the action name
        for j = 1:3
            if i<= 3
                action_idx = top3_snr{i, alg}(j);  % Get index of the top 3 action
            else
                action_idx = top3_user{i-3, alg}(j);  % Get index of the top 3 action
            end
            action_name = models{action_idx};  % Get action name from the model list
            text(top_3_values(j), alg, action_name, 'FontSize', 10, 'FontName', 'Times New Roman', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
        end
    end
    
    % Set axis labels
    xlabel('Probability', 'FontSize', 14, 'FontName', 'Times New Roman');
    yticks(1:5);
    yticklabels({'OMNIS', 'CTO', 'DTS', 'GDO', 'RSS'});  % Algorithm names
    
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
end

% Apply tightfig to remove extra whitespace
tightfig;
