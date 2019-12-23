
clear all

% Load data from file
load voicesTable

%% Initial settings and processing

% Rename table to T for conveinience
T = voices;
clear voices

% Remove centroid since it is a repeat of meanfreq
T(:, {'centroid'}) = [];

% Set "label" to be categorical
T.label = categorical(T.label);

% Get all the variable names and remove the label
Vars = T.Properties.VariableNames;
Vars(end) = [];

% Total data points in each group
N = height(T)/2;

% Get number of variables
numVars = length(Vars);

% Set statistical test parameters
alpha = 0.05;
alpha_bonf = alpha; %/numVars;
num_sub_samples = 50;
idx_sub_sample = randperm(N);
        
% Define group labels and colors for plotting
groups = {'female','male'}; % group labels
cols = 'rb'; % group colors


%% Simple 1D inspection

do_1D_histograms = 1;
if do_1D_histograms
    
    % Clear figure
    figure(100); clf; hold on
    
    is_log_idx = [7 8 14 15 17 18];
    
    logVars = Vars(is_log_idx);
    
    % Initialize criterion measures variable
    criterion = nan(numVars, 4); %
    
    % Loop through all variable and plot malevs female histograms
    for i = 1 : length(Vars)
        
        disp(['Processing ' Vars{i}])
        
        % Prep figure subplot
        ax = subplot(4,5,i);
        % Clear axis
        cla; hold on
        
        % Initialize data variable
        data = nan(N,2);
        
        for j = 1 : length(groups) % female and male
            
            % Get data from table
            data(:,j) = T.(Vars{i})(T.label == groups{j});
            
            if i == 15

                % Flip data
                data(:,j) = 1./data(:,j);
                
            end
            
            % Apply log transform if necessary
            if ismember(Vars{i}, logVars)
                
                idx_zero = data(:,j) == 0;
                if ~isempty(idx_zero) && sum(idx_zero) > 0
                    data(idx_zero,j) = eps; % remove zeros
                end
                data(:,j) = log2(data(:,j));
                
                % Indicate transformed variable by axis color red
                ax.GridColor = 'r';
                ax.XColor = 'r';
                ax.YColor = 'r';

            end
            
            
        end
        
        bins = linspace(min(data(:)), max(data(:)), 50);  % Calculate bins for histograms
        for j = 1 : 2
            % plotting histograms
            histogram(data(:,j),bins, 'FaceColor', cols(j), 'FaceAlpha', 0.5, 'Normalization','probability')
        end
        
        
        %% Statistical significance

        idx_sub_sample = idx_sub_sample(1:num_sub_samples);
        
        % Calculate mean to median ratio as crude normality test
        m_m_ratio_1 = mean(data(:,1)) / median(data(:,1));
        m_m_ratio_2 = mean(data(:,2)) / median(data(:,2));
        
        % Conduct stat test depending on normality
        if (m_m_ratio_1 > 0.95 && m_m_ratio_1 < 1.05)  && (m_m_ratio_2 > 0.95 && m_m_ratio_2 < 1.05)
            
            is_normal = true;
            
            % T-test
            [~, p] = ttest2(data(idx_sub_sample,1), data(idx_sub_sample,2),'Vartype','unequal');
            
        else
            
            is_normal = false;
            
            % Mann-Whitney U test
            p = ranksum(data(idx_sub_sample,1),data(idx_sub_sample,2));
            
        end
        
        % Copy to criterion variable
        criterion(i,1) = p;
        criterion(i,4) = is_normal;
        
        % Check for significance (Bonferroni corrected)
        is_significant = p < alpha_bonf;

        % Rounding fpr plotting purposes
        rounding_factor = 10000;
        p = round(p*rounding_factor)/rounding_factor;
        
        % Sensitivity
        d_prime = abs(mean(data(:,1)) - mean(data(:,2)))/sqrt(0.5*(std(data(:,1)) + std(data(:,2))));
        criterion(i,2) = d_prime; % Copy to criterion variable

        % Rounding fpr plotting purposes
        d_prime  = round(d_prime *rounding_factor)/rounding_factor;
        
        % Divergence
        div = divergence(data(:,1)', data(:,2)');
        criterion(i,3) = div; % Copy to criterion variable
        div  = round(div *rounding_factor)/rounding_factor;
        
        % Update figure labels etc
        xlabel(Vars{i})
        ylabel('Probability')
        grid on; box on
        ylim([0 0.2])
        
        
        % Print test reults onto figure
        if is_normal
            testID = 'TT';
        else
            testID = 'MWU';
        end
        
        if is_significant
            c_sign = 'g';
        else
            c_sign = 'r';
        end
        
        % Define position of text and write to plot
        x = ax.XLim(1) + 0.1*(ax.XLim(2)-ax.XLim(1));
        y = ax.YLim(1) + 0.9*(ax.YLim(2)-ax.YLim(1));
        text(x, y,[testID ': p = ' num2str(p)],'FontSize',9, 'Color', c_sign)
        y = y - 0.15*y;
        text(x, y,['d'' = ' num2str(d_prime)],'FontSize',9)
        y = y - 0.15*y;
        text(x, y,['div = ' num2str(div)],'FontSize',9)
        
    end
    
    
    
    %% Repeat 1D histoframs after sorting according to divergence
    
    [~, divergenceSorting] = sort(criterion(:,3), 'descend');
    Vars = Vars(divergenceSorting);
    
    criterion = criterion(divergenceSorting,:);
    
    figure(101);clf;
    
    % Loop through all variable and plot malevs female histograms
    for i = 1 : length(Vars)
        
        % Prep figure subplot
        ax = subplot(4,5,i);
        % Clear axis
        cla; hold on
        
        % Initialize data variable
        data = nan(N,2);
        
        for j = 1 : length(groups) % female and male
            
            % Get data from table
            data(:,j) = T.(Vars{i})(T.label == groups{j});
            
            if strcmpi(Vars{i},'maxfun') % i == 15
                
                % Flip data
                data(:,j) = 1./data(:,j);
                
            end
            
            % Apply transform if necessary
            if ismember(Vars{i}, logVars)
                
                idx_zero = data(:,j) == 0;
                if ~isempty(idx_zero) && sum(idx_zero) > 0
                    data(idx_zero,j) = eps; % remove zeros
                end
                data(:,j) = log2(data(:,j));
                ax.GridColor = 'r';
                ax.XColor = 'r';
                ax.YColor = 'r';

            end
            
            
        end
        
        bins = linspace(min(data(:)), max(data(:)), 50);  % Calculate bins for histograms
        H = [];
        for j = 1 : 2
            % plotting histograms
            h = histogram(data(:,j),bins, 'FaceColor', cols(j), 'FaceAlpha', 0.5, 'Normalization','probability');
            H = [H h];
        end
        
        
        %% Statistical significance
        p = criterion(i,1);
        is_significant = p < alpha_bonf;
        
        rounding_factor = 10000;
        p = round(p*rounding_factor)/rounding_factor;
        
        % Sensitivity
        d_prime = criterion(i,2);
        d_prime  = round(d_prime *rounding_factor)/rounding_factor;
        
        % Divergence
        div = criterion(i,3);
        div  = round(div *rounding_factor)/rounding_factor;
        
        % Update figure labels etc
        xlabel(Vars{i})
        ylabel('Probability')
        grid on; box on
        
        ylim([0 max(0.2, max([H(1).Values H(2).Values])*1.05)])
        
        
        % Print test reults onto figure
        is_normal = criterion(i,4);
        if is_normal
            testID = 'TT';
        else
            testID = 'MWU';
        end
                
        if is_significant
            c_sign = 'g';
        else
            c_sign = 'r';
        end
        
        x = ax.XLim(1) + 0.1*(ax.XLim(2)-ax.XLim(1));
        y = ax.YLim(1) + 0.9*(ax.YLim(2)-ax.YLim(1));
        text(x, y,[testID ': p = ' num2str(p)],'FontSize',9, 'Color', c_sign)
        y = y - 0.15*y;
        text(x, y,['d'' = ' num2str(d_prime)],'FontSize',9)
        y = y - 0.15*y;
        text(x, y,['div = ' num2str(div)],'FontSize',9)
        
    end
    
    
    
    %% Repeat after removing all non-significant and low divergence features
    
    figure(102);clf;
    
    % Find non-significant features
    did_not_make_it = criterion(:,1) > alpha_bonf | criterion(:,3) < 2;
    
    % Remove non-significant features and criterion values
    Vars(did_not_make_it) = [];
    criterion(did_not_make_it,:) = [];
    numVars = length(Vars);

    % Prep figure subplot
    if numVars <= 5
        numRows = 1;
    elseif numVars <= 10
        numRows = 2;
    elseif numVars <= 5
        numRows = 3;
    else
        numRows = 4;
        
    end
    
    
    % Loop through all variable and plot malevs female histograms
    for i = 1 : length(Vars)
        
 
        ax = subplot(numRows,5,i);

        % Clear axis
        cla; hold on
        
        % Initialize data variable
        data = nan(N,2);
        
        for j = 1 : length(groups) % female and male
            
            % Get data from table
            data(:,j) = T.(Vars{i})(T.label == groups{j});
            
            if strcmpi(Vars{i},'maxfun') % i == 15
                
                % Flip data
                data(:,j) = 1./data(:,j);
                
            end
            
            % Apply transform if necessary
            if ismember(Vars{i}, logVars)
                
                idx_zero = data(:,j) == 0;
                if ~isempty(idx_zero) && sum(idx_zero) > 0
                    data(idx_zero,j) = eps; % remove zeros
                end
                data(:,j) = log2(data(:,j));
                ax.GridColor = 'r';
                ax.XColor = 'r';
                ax.YColor = 'r';

            end
            
            
        end
        
        bins = linspace(min(data(:)), max(data(:)), 50);  % Calculate bins for histograms
        H = [];
        for j = 1 : 2
            % plotting histograms
            h = histogram(data(:,j),bins, 'FaceColor', cols(j), 'FaceAlpha', 0.5, 'Normalization','probability');
            H = [H h];
        end
        
        
        %% Statistical significance
        p = criterion(i,1);
        is_significant = p < alpha_bonf;
        
        rounding_factor = 10000;
        p = round(p*rounding_factor)/rounding_factor;
        
        % Sensitivity
        d_prime = criterion(i,2);
        d_prime  = round(d_prime *rounding_factor)/rounding_factor;
        
        % Divergence
        div = criterion(i,3);
        div  = round(div *rounding_factor)/rounding_factor;
        
        % Update figure labels etc
        xlabel(Vars{i})
        ylabel('Probability')
        grid on; box on
        
        ylim([0 max(0.2, max([H(1).Values H(2).Values])*1.05)])
        
        
        % Print test reults onto figure
        is_normal = criterion(i,4);
        if is_normal
            testID = 'TT';
        else
            testID = 'MWU';
        end
                
        if is_significant
            c_sign = 'g';
        else
            c_sign = 'r';
        end
        
        x = ax.XLim(1) + 0.1*(ax.XLim(2)-ax.XLim(1));
        y = ax.YLim(1) + 0.9*(ax.YLim(2)-ax.YLim(1));
        text(x, y,[testID ': p = ' num2str(p)],'FontSize',9, 'Color', c_sign)
        y = y - 0.15*y;
        text(x, y,['d'' = ' num2str(d_prime)],'FontSize',9)
        y = y - 0.15*y;
        text(x, y,['div = ' num2str(div)],'FontSize',9)
        
    end    
    
    
end

%% Calculate and plot divergence

doDivergence = 1;

D = nan(numVars, numVars);
if doDivergence
    
    for i = 1 : numVars
        for j = 1 : numVars
            D(i,j) = divergence(T{T.label == groups{1}, [Vars(i) Vars(j)]}', T{T.label == groups{2}, [Vars(i) Vars(j)]}');
        end
    end
    
    plotMatrix(D, Vars, 200, 'Divergence', true);
    
end

%% Calculate and plot correlations

doCorrelations = 1;

if doCorrelations
    
    % Retrieve data 
    r_data = T{:,Vars};
    % Perform correlations comparing all variable pairs
    R = corrcoef(r_data); 
    R = abs(R); % Take absolute values since we don't care about the direction
    
    % Plot results
    [~, ax] = plotMatrix(R, Vars, 201, 'Anti-correlation');
    
end


%% Calculate and plot overall optimal feature combinations

doDivCorrOpti = 1;

if doDivCorrOpti

    % Remove Inf values from divergence matrix
    D(D == Inf) = min(min(D));
    
    % Normalize divergence values
    D = (D - min(min(D)))/(max(max(D)) - min(min(D)));
    
    % Multiply inverse of correlation with divergence
    M = (1-R) .* D;
    
    % Plot results
    plotMatrix(M, Vars, 203, 'Optimal Divergence-Correlation combo');
    
end


% Below is dormant for now
% 
% % Clear figure
% figure(100); clf; hold on
% 
% do_scatter = 0;
% if do_scatter
%     
%     Vars = sensitivitySort(T);
%     
%     % Loop through all variable and plot malevs female histograms
%     for i = 1 : length(Vars)-1
%         
%         
%         %% Data retrieval
%         
%         % Get male and female data from table
%         male_v1 = T.(Vars{i})(T.label == 'male');
%         male_v2 = T.(Vars{i+1})(T.label == 'male');
%         
%         female_v1 = T.(Vars{i})(T.label == 'female');
%         female_v2 = T.(Vars{i+1})(T.label == 'female');
%         
%         
%         %% Plotting
%         
%         % plotting histograms
%         subplot(1,2,1); cla; hold on
%         scatter(male_v1, male_v2,'bx')
%         scatter(female_v1, female_v2,'ro')
%         
%         % Update figure labels etc
%         legend({'Male','Female'}); grid on; box on
%         xlabel(Vars{i})
%         ylabel(Vars{i+1})
%         
%         subplot(1,2,2); cla; hold on
%         histogram2(male_v1, male_v2,50,'Normalization','probability','DisplayStyle','tile','FaceColor','flat')
%         histogram2(female_v1, female_v2,50,'Normalization','probability','DisplayStyle','tile','FaceColor','flat')
%         
%         % Update figure labels etc
%         legend; grid on; box on
%         xlabel(Vars{i})
%         ylabel(Vars{i+1})
%         
%         
%     end
% end
% 
% disp('DONE!')
% 
% sorted = sensitivitySort(T);


%% Funtions

function [h, ax] = plotMatrix(Mat, Vars, fignum, plotTitle, doLog)

if nargin < 5
    doLog = false;
end

if doLog
    % Perform log transform for better visialization
    Mat = log2(Mat);
    
    if any(Inf)
        idx = Mat == Inf;
        Mat(idx) = min(min(Mat));
    end
    
end

% Create figure
h = figure(fignum); clf;

% Get number of variables
numVars = length(Vars);

% Calculate X and Y values
[X, Y] = meshgrid(1:numVars+1, 1:numVars+1);

% Add zeroes for plotting
Mat = [Mat zeros(length(Mat(:,1)),1)];
Mat = [Mat; zeros(1, length(Mat(1,:)))];

% Plot surface plot
surface(X,Y,Mat)

% Set view
view(2)

% Define color map and bar
colorbar
colormap default

% Define axis
Min = min(min(Mat));
Max = max(max(Mat));
caxis([Min Max])

% Set X and Y ticks
xticks(1.5 : 1 : numVars + 0.5)
xticklabels(Vars)
xtickangle(90)

yticks(1.5 : 1 : numVars + 0.5)
yticklabels(Vars)
ytickangle(0)

% Set title, labels, and limits
title(plotTitle)
xlabel('Variable', 'FontSize',14)
ylabel('Variable', 'FontSize',14)
xlim([1 numVars+1])
ylim([1 numVars+1])

% Get the axis handle for function output
ax = gca;

end
