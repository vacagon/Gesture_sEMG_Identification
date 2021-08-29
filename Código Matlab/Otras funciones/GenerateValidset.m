rng('default');

trainsplit = 0.7;

%% DEEP LEARNING DATASET

% Mano Abierta

A = buffer(MA.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(MA.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(MA.Normalized(:,3), 200, 50, 'nodelay');
ManoAbierta = cat(3, A, B, C);
[~,nObsMA,~] = size(ManoAbierta);
MA_label = repmat(4, nObsMA, 1);

% Muñeca Flexionada

A = buffer(MF.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(MF.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(MF.Normalized(:,3), 200, 50, 'nodelay');
MunecaFlexionada = cat(3, A, B, C);
[~,nObsMF,~] = size(MunecaFlexionada);
MF_label = repmat(3, nObsMF, 1);

% Mano Relajada

A = buffer(MR.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(MR.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(MR.Normalized(:,3), 200, 50, 'nodelay');
ManoRelajada = cat(3, A, B, C);
[~,nObsMR,~] = size(ManoRelajada);
MR_label = ones(nObsMR, 1);

% Puño Cerrado

A = buffer(PC.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(PC.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(PC.Normalized(:,3), 200, 50, 'nodelay');
PunoCerrado = cat(3, A, B, C);
[~,nObsPC,~] = size(PunoCerrado);
PC_label = repmat(2, nObsPC, 1);

% Muñeca Extendida

A = buffer(ME.Normalized(:,1), 200, 50, 'nodelay');
B = buffer(ME.Normalized(:,2), 200, 50, 'nodelay');
C = buffer(ME.Normalized(:,3), 200, 50, 'nodelay');
MunecaExtendida = cat(3, A, B, C);
[~,nObsME,~] = size(MunecaExtendida);
ME_label = repmat(5, nObsME, 1);

clear A B C;

X = [ManoRelajada, PunoCerrado, MunecaFlexionada, ManoAbierta,...
    MunecaExtendida];
X = permute(X, [2,1,3]);
Y = [MR_label; PC_label; MF_label; MA_label; ME_label];
X = num2cell(X, [2 3]);
Y = categorical(Y);

clear MA_label MF_label MR_label PC_label ME_label ...
nObsMA nObsMF nObsMR nObsPC nObsME;

%% MACHINE LEARNING DATASET

% Feature Offline

% Mano Relajada

MR_Features = [feature_offline(MR.Filtered(:,1), MR.Normalized(:,1));...
     feature_offline(MR.Filtered(:,2), MR.Normalized(:,2));...
     feature_offline(MR.Filtered(:,3), MR.Normalized(:,3))];
[~,n] = size(MR_Features);
MR_Label = 1;
MR_Label = repmat(MR_Label, n, 1);

% Puño Cerrado

PC_Features = [feature_offline(PC.Filtered(:,1), PC.Normalized(:,1));...
     feature_offline(PC.Filtered(:,2), PC.Normalized(:,2));...
     feature_offline(PC.Filtered(:,3), PC.Normalized(:,3))];
[~,n] = size(PC_Features);
PC_Label = 2;
PC_Label = repmat(PC_Label, n, 1);

% Muñeca Flexionada

MF_Features = [feature_offline(MF.Filtered(:,1), MF.Normalized(:,1));...
     feature_offline(MF.Filtered(:,2), MF.Normalized(:,2));...
     feature_offline(MF.Filtered(:,3), MF.Normalized(:,3))];
[~,n] = size(MF_Features);
MF_Label = 3;
MF_Label = repmat(MF_Label, n, 1); 

% Mano Abierta

MA_Features = [feature_offline(MA.Filtered(:,1), MA.Normalized(:,1));...
     feature_offline(MA.Filtered(:,2), MA.Normalized(:,2));...
     feature_offline(MA.Filtered(:,3), MA.Normalized(:,3))];
[~,n] = size(MA_Features);
MA_Label = 4;
MA_Label = repmat(MA_Label, n, 1); 

% Muñeca Extendida

ME_Features = [feature_offline(ME.Filtered(:,1), ME.Normalized(:,1));...
     feature_offline(ME.Filtered(:,2), ME.Normalized(:,2));...
     feature_offline(ME.Filtered(:,3), ME.Normalized(:,3))];
[~,n] = size(ME_Features);
ME_Label = 5;
ME_Label = repmat(ME_Label, n, 1); 

Features = [MR_Features, PC_Features, MF_Features, MA_Features, ME_Features]';
Label = [MR_Label; PC_Label; MF_Label; MA_Label; ME_Label];
Label = categorical(Label);

Dataset = table(Features, Label);

clear MR PC MF MA ME MR_Features PC_Features MF_Features MA_Features ME_Features...
    MR_Label PC_Label MF_Label MA_Label ME_Label n;