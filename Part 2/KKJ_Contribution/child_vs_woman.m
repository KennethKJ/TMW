
% Clear slate
clear all

% Load table with extrated features from file
load FeatureTable

% Load labels from txt file
file_path = 'C:\Users\Research\Google Drive\TMW Working dir\';
filename = 'Label Track2.txt';
Labels_Audio1 = importLabelFile([file_path filename]);

% Get variable names of features
vars = fT.Properties.VariableNames(2:end); % Grab all variables except time

% Get class names from label table
class = unique(Labels_Audio1.Class);

for i = 1 : length(vars)
    
    for c = class
        
        data = fT.
        
        
    end
end


clc