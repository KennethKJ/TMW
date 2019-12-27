clc

totalTable = [];

% Loop through all audio files

for f = 1 : 3
    
    filename = ['audio' num2str(f) '.wav'];
    
    % Get audio signal and sample rate
    [signal, fs] = audioread(filename);
    
    
    %% Settings
    
    % Define window size in seconds (1024 samples)
    win = 1024/fs;
    
    % Define step size;
    step = win * 0.5;
    
    %% ************* Feature extraction *************************
    
    %% Using ToolBox 1
    
    % Calculate features:
    features1 = stFeatureExtraction(signal, fs, win, step);
    
    %% Construct feature table from array
    
    fT1 = array2table(features1');
    
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
    
    fT1.Properties.VariableNames(1:8) = featureNames(1:8);
    
    % Name the MFCC columns
    for i = 9 : 21
        fT1.Properties.VariableNames{i} = ['MFCC_' num2str(i-9)];
    end
    
    fT1.Properties.VariableNames(22:23) = featureNames(10:11);
    
    % Name the Chroma columns
    for i = 24 : size(features1,1)
        fT1.Properties.VariableNames{i} = ['chroma_vector_' num2str(i-24)];
    end
    
    % Add time to the table
    t = (step : step : length(signal)/fs-step)';  % Calculate time vector
    fT1.TimeStep = t;  % Add time vector to table
    fT1 = [fT1(:, end) fT1(:, 1:end-1)]; % Bring time column to be first
    
    
    %% Using ToolBox 2
    
    % Cell array over features to extract
    FTs = {
        'mfcc';
        'mfccDelta';
        'mfccDeltaDelta';
        'spectralEntropy';
        'spectralFlux'
        };
    
    
    % Initialize audio feature extractor
    aFE = audioFeatureExtractor();
    
    % Set sample rate
    aFE.SampleRate = fs;
    
    % Set flags of features we want to extract
    for i = 1 : length(FTs)
        
        aFE.(FTs{i}) = true;
        
    end
    
    % Extract selected features
    features2 = extract(aFE,signal);
    
    % Get the index of each feature
    idx = info(aFE);
    
    % Convert extracted features to table
    fT2 = array2table(features2);
    
    % Initialize variable name cell
    varNames = {};
    
    % Compute all variable names
    for i = 1 : length(FTs)
        currentVar = FTs{i};
        for j = 1 : length(idx.(currentVar))
            
            if length(idx.(currentVar)) == 1
                varNames{end+1} = currentVar;
            else
                varNames{end+1} = [currentVar '_' num2str(j)];
            end
        end
    end
    
    % Set table variable names
    fT2.Properties.VariableNames = varNames;
    
    % Add filename (for reference)
    fT2.filename(1: height(fT2)) = categorical({filename});
    
    %% Combining results
    
    % Make cell array over features to take from toolbox 1
    from_fT1 = {
        'TimeStep';
        'zcr';
        };
    
    % Combine selected toolbox 1 features with toolbox 2 features
    totalTable = [totalTable; fT1(:,from_fT1) fT2];
    
    
end

% Save table to file
save('FeatureTable.mat','totalTable')

% Write table to CSV
writetable(totalTable, 'features.csv');

disp('Done')



