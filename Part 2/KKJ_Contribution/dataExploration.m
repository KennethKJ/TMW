
clear all

% Load data from file
load voicesTable

% Rename table to T for conveinience
T = voices;
clear voices

% Set "label" to be categorical
T.label = categorical(T.label);

% Get all the variable names and remove the label
Vars = T.Properties.VariableNames;
Vars(end) = [];

% Group labels
groups = {'female','male'}; % group labels
cols = 'rb'; % group colors

% Total data points in each group
N = height(T)/2;

% Clear figure
figure(100); clf; hold on
ax = gca;

%% Simple 1D inspection

do_1D_histograms =0;
if do_1D_histograms
    % Loop through all variable and plot malevs female histograms
    for i = 1 : length(Vars)
        
        % Clear axis
        cla; hold on
        
        % Initialize data variable
        data = nan(N,2);
        
        for j = 1 : length(groups) % female and male
            
            % Get data from table
            data(:,j) = T.(Vars{i})(T.label == groups{j});
            
            % plotting histograms
            histogram(data(:,j),50, 'FaceColor', cols(j), 'FaceAlpha', 0.5)
            
        end
        
        
        %% Statistical significance
        
        % T-test
        [~, p_t] = ttest2(data(:,1), data(:,2),'Vartype','unequal');
        % Mann-Whitney U test
        p_mw = ranksum(data(:,1),data(:,2));
        % Sensitivity
        d_prime = abs(mean(data(:,1)) - mean(data(:,2)))/sqrt(0.5*(std(data(:,1)) + std(data(:,2))));
        
        
        % plotting mean values
        for j = 1 : length(groups)
            plot([mean(data(:,j)) mean(data(:,j))],[0 ax.YLim(2)],cols(j), 'LineWidth',3)
            plot([median(data(:,j)) median(data(:,j))],[0 ax.YLim(2)],[cols(j) '--'], 'LineWidth',3)
        end
        
        
        % Update figure labels etc
        legend(groups)
        xlabel(Vars{i})
        ylabel('Frequency')
        
        % Print test reults onto figure
        x = ax.XLim(1) + 0.1*(ax.XLim(2)-ax.XLim(1));
        y = ax.YLim(1) + 0.9*(ax.YLim(2)-ax.YLim(1));
        text(x, y,['T-test p-value = ' num2str(p_t)],'FontSize',16)
        y = y - 0.05*y;
        text(x, y,['d prime = ' num2str(d_prime)],'FontSize',16)
        y = y - 0.05*y;
        text(x, y,['MW U p-value = ' num2str(p_mw)],'FontSize',16)
        
        
    end
end


% Clear figure
figure(100); clf; hold on

do_scatter = 1;
if do_scatter
    % Loop through all variable and plot malevs female histograms
    for i = 1 : length(Vars)-1
        
        
        %% Data retrieval
        
        % Get male and female data from table
        male_v1 = T.(Vars{i})(T.label == 'male');
        male_v2 = T.(Vars{i+1})(T.label == 'male');
        
        female_v1 = T.(Vars{i})(T.label == 'female');
        female_v2 = T.(Vars{i+1})(T.label == 'female');
        
        
        %% Plotting
        
        % plotting histograms
        subplot(1,2,1); cla; hold on
        scatter(male_v1, male_v2,'bx')
        scatter(female_v1, female_v2,'ro')
        
        % Update figure labels etc
        legend({'Male','Female'}); grid on; box on
        xlabel(Vars{i})
        ylabel(Vars{i+1})
        
        subplot(1,2,2); cla; hold on
        histogram2(male_v1, male_v2,50,'Normalization','probability','DisplayStyle','tile','FaceColor','flat')
        histogram2(female_v1, female_v2,50,'Normalization','probability','DisplayStyle','tile','FaceColor','flat')
        
        % Update figure labels etc
        legend; grid on; box on
        xlabel(Vars{i})
        ylabel(Vars{i+1})
        
        
    end
end

disp('DONE!')



