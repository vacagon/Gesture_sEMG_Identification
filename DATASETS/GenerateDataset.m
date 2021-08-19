clear variables;
clc;

load('datos_gestos_combinados.mat')
rng('default');

trainsplit = 0.7;

%% DEEP LEARNING DATASET

% Mano Abierta

A = buffer(MA.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(MA.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(MA.Normalized(:,3), 200, 50, 'nodelay');
MA.Normalized = cat(3, A, B, C);
[~,nObsMA,~] = size(MA.Normalized);
MA_label = repmat(4, nObsMA, 1);

% Muñeca Flexionada

A = buffer(MF.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(MF.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(MF.Normalized(:,3), 200, 50, 'nodelay');
MF.Normalized = cat(3, A, B, C);
[~,nObsMF,~] = size(MF.Normalized);
MF_label = repmat(3, nObsMF, 1);

% Mano Relajada

A = buffer(MR.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(MR.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(MR.Normalized(:,3), 200, 50, 'nodelay');
MR.Normalized = cat(3, A, B, C);
[~,nObsMR,~] = size(MR.Normalized);
MR_label = ones(nObsMR, 1);

% Puño Cerrado

A = buffer(PC.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(PC.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(PC.Normalized(:,3), 200, 50, 'nodelay');
PC.Normalized = cat(3, A, B, C);
[~,nObsPC,~] = size(PC.Normalized);
PC_label = repmat(2, nObsPC, 1);

% Muñeca Extendida

A = buffer(ME.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(ME.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(ME.Normalized(:,3), 200, 50, 'nodelay');
ME.Normalized = cat(3, A, B, C);
[~,nObsME,~] = size(ME.Normalized);
ME_label = repmat(5, nObsME, 1);

clear A B C;

% Training and Validation Datasets

Gestures = [MR.Normalized, PC.Normalized, MF.Normalized, MA.Normalized, ME.Normalized];
Gestures = permute(Gestures, [2,1,3]);
Label = [MR_label; PC_label; MF_label; MA_label; ME_label];
cv = cvpartition(Label, 'HoldOut', 1 - trainsplit);

XTrain = Gestures(cv.training(1), :, :);
XTrain = num2cell(XTrain, [2 3]);
YTrain = categorical(Label(cv.training(1), :, :));

XValid = Gestures(cv.test(1), :, :);
XValid = num2cell(XValid, [2 3]);
YValid = categorical(Label(cv.test(1), :, :));
clear MA MR PC MF ME MA_label MF_label MR_label PC_label ME_label ...
    Gestures Label cv;

numObservations = nObsMA + nObsMF + nObsMR + nObsPC + nObsME;
clear nObsMA nObsMF nObsMR nObsPC nObsME;

%% MACHINE LEARNING DATASET

% Feature Offline

load('datos_gestos_combinados.mat');

% Mano Relajada

MR_Features = [feature_offline(MR.Filtered(:,1), MR.Normalized(:,1));...
     feature_offline(MR.Filtered(:,2), MR.Normalized(:,2));...
     feature_offline(MR.Filtered(:,3), MR.Normalized(:,3))];
[~,n] = size(MR_Features);
MR_Label = "Mano Relajada";
MR_Label = repmat(MR_Label, n, 1);

% Puño Cerrado

PC_Features = [feature_offline(PC.Filtered(:,1), PC.Normalized(:,1));...
     feature_offline(PC.Filtered(:,2), PC.Normalized(:,2));...
     feature_offline(PC.Filtered(:,3), PC.Normalized(:,3))];
[~,n] = size(PC_Features);
PC_Label = "Puño Cerrado";
PC_Label = repmat(PC_Label, n, 1);

% Muñeca Flexionada

MF_Features = [feature_offline(MF.Filtered(:,1), MF.Normalized(:,1));...
     feature_offline(MF.Filtered(:,2), MF.Normalized(:,2));...
     feature_offline(MF.Filtered(:,3), MF.Normalized(:,3))];
[~,n] = size(MF_Features);
MF_Label = "Muñeca Flexionada";
MF_Label = repmat(MF_Label, n, 1); 

% Mano Abierta

MA_Features = [feature_offline(MA.Filtered(:,1), MA.Normalized(:,1));...
     feature_offline(MA.Filtered(:,2), MA.Normalized(:,2));...
     feature_offline(MA.Filtered(:,3), MA.Normalized(:,3))];
[~,n] = size(MA_Features);
MA_Label = "Mano Abierta";
MA_Label = repmat(MA_Label, n, 1); 

% Muñeca Extendida

ME_Features = [feature_offline(ME.Filtered(:,1), ME.Normalized(:,1));...
     feature_offline(ME.Filtered(:,2), ME.Normalized(:,2));...
     feature_offline(ME.Filtered(:,3), ME.Normalized(:,3))];
[~,n] = size(ME_Features);
ME_Label = "Muñeca Extendida";
ME_Label = repmat(ME_Label, n, 1); 

% Training and Validation DGatasets

Features = [MR_Features, PC_Features, MF_Features, MA_Features, ME_Features]';
Label = [MR_Label; PC_Label; MF_Label; MA_Label; ME_Label];
Label = categorical(Label);

cv = cvpartition(Label, 'HoldOut', 1 - trainsplit);
Dataset = table(Features, Label);
Trainset = table(Features(cv.training,:), Label(cv.training));
Validset = table(Features(cv.test,:), Label(cv.test));

Validset.Properties.VariableNames{1} = 'Features';
Validset.Properties.VariableNames{2} = 'Label';
Trainset.Properties.VariableNames{1} = 'Features';
Trainset.Properties.VariableNames{2} = 'Label';

clear MR PC MF MA ME MR_Features PC_Features MF_Features MA_Features ME_Features...
    MR_Label PC_Label MF_Label MA_Label ME_Label n cv;

%% Save generated datasets

save('C:\TFG\Matlab_Files\RESULTS\DL_data_valid2.mat', 'XTrain', 'XValid', 'YTrain', 'YValid');
save('C:\TFG\Matlab_Files\RESULTS\ML_datavalid2.mat', 'Trainset', 'Validset', 'Dataset');
