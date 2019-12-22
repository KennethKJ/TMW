%% Settings

% Define window size
win = 0.1;

% Define step size;
step = win * 0.5;


%% Feature extraction

doFeatureExtraction = 1;
if doFeatureExtraction

    % Get audio signal and sample rate
    [signal, fs] = audioread('audio1.wav');
    
    % Calculate features:
    Features = stFeatureExtraction(signal, fs, win, step);
    
    %% Construct feature table from array
    
    fT = array2table(Features');
    
    % Name all the columns
    featureNames = {... % Define feature names
        'zcr', ...
        'energy', ...
        'energy_entropy', ...
        'spectral_centroid', ...
        'spectral_centroid_spread', ...
        'spectral_entropy', ...
        'spectral_flux', ...
        'spectral_rolloff', ...
        'MFCCs', ...  % 9:21
        'harmonic_ratio', ...
        'F0', ...
        'chroma_vector'}; % 23+1:23+12
    
    fT.Properties.VariableNames(1:8) = featureNames(1:8);
    
    % Name the MFCC columns
    for i = 9 : 21
        fT.Properties.VariableNames{i} = ['MFCC_' num2str(i-9)];
    end
    
    fT.Properties.VariableNames(22:23) = featureNames(10:11);
    
    % Name the Chroma columns
    for i = 24 : size(Features,1)
        fT.Properties.VariableNames{i} = ['chroma_vector_' num2str(i-24)];
    end
    
    % Add time to the table
    t = (step : step : length(signal)/fs-step)';  % Calculate time vector
    fT.TimeStep = t;  % Add time vector to table
    fT = [fT(:, end) fT(:, 1:end-1)]; % Bring time column to be first
    
    
    % Save table to file
    save('FeatureTable.mat','fT')
    
end

%% Plot histograms
% 
% t = step : step : length(signal)/fs-step;
% figure(200); clf
% 
% for m = 0 : 12
%     
%     currentFeature = ['MFCC_' num2str(m)];
%     
%     data = fT.(currentFeature);
%     
%     
%     subplot(2,6, m+1); cla; hold on
%     histogram(data, 50,'Normalization','probability')
%     
%     
%     xlabel(currentFeature)
%     
%     
% end



figure(300); clf
Vars = fT.Properties.VariableNames;
for i = 1 : length(Vars)
    
    currentFeature = Vars{i};
    
    data = fT.(currentFeature);
    
    
    %     subplot(2,6, m+1);
    cla; hold on
    histogram(data, 150,'Normalization','probability')
    
    
    xlabel(currentFeature)
    
    
end







