clc;
clear all;

% Network parameters

Network_parameters;

%% ENTRENAMIENTO DE LOS MODELOS CON LOS DATOS OBTENIDOS EL 08/07/2021

load('datos_gestos_0807.mat');

GenerateDataset;

% DEEP LEARNING

        % LSTM + CNN

layersLSTMCNN = [...
    sequenceInputLayer(inputSize, 'Name', 'input')
    sequenceFoldingLayer('Name', 'fold')
    convolution2dLayer(filterSize, numFilters, 'Name', 'conv')
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayer(numHiddenUnits,'OutputMode','last','Name','lstm')
    dropoutLayer(dropoutProbability, 'Name', 'doutprob')
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

lgraphLSTMCNN = layerGraph(layersLSTMCNN);
lgraphLSTMCNN = connectLayers(lgraphLSTMCNN,'fold/miniBatchSize','unfold/miniBatchSize');

optionsLSTMCNN = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience',5, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',true, ...
    'VerboseFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.1);

accLSTMCNN = NaN(numIter, 1);
for i = 1:numIter
    [BestNetLSTMCNN(i), info] = trainNetwork(XTrain,YTrain,lgraphLSTMCNN,optionsLSTMCNN);
    accLSTMCNN(i) = info.FinalValidationAccuracy;
end
[BestAcc(1),n] = max(accLSTMCNN);
BestAcc(1) = BestAcc(1)/100;
netLSTMCNN = BestNetLSTMCNN(n);
YPredLSTMCNN = classify(netLSTMCNN, XValid, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTMCNN = confusionmat(YValid, YPredLSTMCNN);
clear BestNetLSTMCNN YPredLSTMCNN;

        % Long Short-Term Memory

optionsLSTM = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience',5, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',true, ...
    'VerboseFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.1);

layersLSTM = [ ...
    sequenceInputLayer(inputSize)
    flattenLayer
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(dropoutProbability)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

accLSTM = NaN(numIter, 1);
for i = 1:numIter
    [BestNetLSTM(i), info] = trainNetwork(XTrain,YTrain, layersLSTM, optionsLSTM);
    accLSTM(i) = info.FinalValidationAccuracy;
end
[BestAcc(2),n] = max(accLSTM);
BestAcc(2) = BestAcc(2)/100;
netLSTM = BestNetLSTM(n);
YPredLSTM = classify(netLSTM, XValid, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTM = confusionmat(YValid, YPredLSTM);
clear BestNetLSTM YPredLSTM;

        % Gated Recurrent Unit

layersGRU = [ ...
    sequenceInputLayer(inputSize)
    flattenLayer
    gruLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(dropoutProbability)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

optionsGRU = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'ValidationPatience',5, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency', 20, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',true, ...
    'VerboseFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.1);

accGRU = NaN(numIter, 1);
for i = 1:numIter
    [BestNetGRU(i), info] = trainNetwork(XTrain,YTrain, layersGRU, optionsGRU);
    accGRU(i) = info.FinalValidationAccuracy;
end
[BestAcc(3),n] = max(accGRU);
BestAcc(3) = BestAcc(3)/100;
netGRU = BestNetGRU(n);
YPredGRU = classify(netGRU, XValid, 'MiniBatchSize', miniBatchSize);
ConfMatrix_GRU = confusionmat(YValid, YPredGRU);
clear BestNetGRU YPredGRU;


    % MACHINE LEARNING

accSVM = NaN(numIter, 1);
for j = 1:numIter
    [BestSVM(j), accSVM(j)] = trainSVM(Trainset);
end
[BestAcc(4), n] = max(accSVM);
SVM = BestSVM(n);
SVM_Pred = SVM.predictFcn(Validset);
ConfMatrix_SVM = confusionmat(Validset.Label, SVM_Pred);
clear BestSVM SVM_Pred;

accDA = NaN(numIter, 1);
for j = 1:numIter
    [BestDA(j), accDA(j)] = trainDA(Trainset);
end
[BestAcc(5), n] = max(accDA);
DA = BestDA(n);
DA_Pred = DA.predictFcn(Validset);
ConfMatrix_DA = confusionmat(Validset.Label, DA_Pred);
clear BestDA DA_Pred;

accKNN = NaN(numIter, 1);
for j = 1:numIter
    [BestKNN(j), accKNN(j)] = trainKNN(Trainset);
end
[BestAcc(6), n] = max(accKNN);
KNN = BestKNN(n);
KNN_Pred = KNN.predictFcn(Validset);
ConfMatrix_KNN = confusionmat(Validset.Label, KNN_Pred);
clear BestKNN KNN_Pred;

accNBC = NaN(numIter, 1);
for j = 1:numIter
    [BestNBC(j), accNBC(j)] = trainNBC(Trainset);
end
[BestAcc(7), n] = max(accNBC);
NBC = BestNBC(n);
NBC_Pred = NBC.predictFcn(Validset);
ConfMatrix_NBC = confusionmat(Validset.Label, NBC_Pred);
clear BestNBC NBC_Pred;

acc = [accLSTMCNN, accLSTM, accGRU, accSVM, accNBC, accKNN, accDA];
for i = 1:3
   acc(:,i) = acc(:,i)./100; 
end
clear accLSTMCNN accLSTM accGRU accSVM accNBC accKNN accDA;

clear n j;

% PLOT

h(1) = figure;
ConfMatrix_LSTMCNN = confusionchart(ConfMatrix_LSTMCNN, classLabels);
ConfMatrix_LSTMCNN.RowSummary = 'row-normalized';
title('CNN+LSTM');

h(2) = figure;
ConfMatrix_LSTM = confusionchart(ConfMatrix_LSTM, classLabels);
ConfMatrix_LSTM.RowSummary = 'row-normalized';
title('LSTM');

h(3) = figure;
ConfMatrix_GRU = confusionchart(ConfMatrix_GRU, classLabels);
ConfMatrix_GRU.RowSummary = 'row-normalized';
title('GRU');

h(4) = figure;
ConfMatrix_SVM = confusionchart(ConfMatrix_SVM, classLabels);
ConfMatrix_SVM.RowSummary = 'row-normalized';
title('SVM');

h(5) = figure;
ConfMatrix_NBC = confusionchart(ConfMatrix_NBC, classLabels);
ConfMatrix_NBC.RowSummary = 'row-normalized';
title('NBC');

h(6) = figure;
ConfMatrix_KNN = confusionchart(ConfMatrix_KNN, classLabels);
ConfMatrix_KNN.RowSummary = 'row-normalized';
title('KNN');

h(7) = figure;
ConfMatrix_DA = confusionchart(ConfMatrix_DA, classLabels);
ConfMatrix_DA.RowSummary = 'row-normalized';
title('DA');

h(8) = figure;
boxplot(acc);

% SAVE RESULTS

save C:\TFG\Matlab_Files\Resultados1.mat acc BestAcc DA KNN NBC SVM netGRU netLSTM netLSTMCNN;

savefig(h, 'C:\TFG\Matlab_Files\ResultPlots1.fig');

clear all;
close all;

%% EVALUACIÓN DE LOS MODELOS ENTRENADOS CON LAS SEÑALES DEL 26/07/2021 CON LAS SEÑALES DEL 03/08/2021

% Network parameters

Network_parameters;

load('datos_gestos_0308');
load('Resultados1.mat');

GenerateValidset;

% CNN + LSTM

YPredLSTMCNN = classify(netLSTMCNN, X, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTMCNN = confusionmat(Y, YPredLSTMCNN);

% LSTM

YPredLSTM = classify(netLSTM, X, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTM = confusionmat(Y, YPredLSTM);

% GRU

YPredGRU = classify(netGRU, X, 'MiniBatchSize', miniBatchSize);
ConfMatrix_GRU = confusionmat(Y, YPredGRU);

% SVM

SVM_Pred = SVM.predictFcn(Dataset);
ConfMatrix_SVM = confusionmat(Dataset.Label, SVM_Pred);

% NBC

NBC_Pred = NBC.predictFcn(Dataset);
ConfMatrix_NBC = confusionmat(Dataset.Label, NBC_Pred);

% KNN

KNN_Pred = KNN.predictFcn(Dataset);
ConfMatrix_KNN = confusionmat(Dataset.Label, KNN_Pred);

% DA

DA_Pred = DA.predictFcn(Dataset);
ConfMatrix_DA = confusionmat(Dataset.Label, DA_Pred);

% PLOTS

h(1) = figure;
ConfMatrix_LSTMCNN = confusionchart(ConfMatrix_LSTMCNN, classLabels);
ConfMatrix_LSTMCNN.RowSummary = 'row-normalized';
title('CNN+LSTM');

h(2) = figure;
ConfMatrix_LSTM = confusionchart(ConfMatrix_LSTM, classLabels);
ConfMatrix_LSTM.RowSummary = 'row-normalized';
title('LSTM');

h(3) = figure;
ConfMatrix_GRU = confusionchart(ConfMatrix_GRU, classLabels);
ConfMatrix_GRU.RowSummary = 'row-normalized';
title('GRU');

h(4) = figure;
ConfMatrix_SVM = confusionchart(ConfMatrix_SVM, classLabels);
ConfMatrix_SVM.RowSummary = 'row-normalized';
title('SVM');

h(5) = figure;
ConfMatrix_NBC = confusionchart(ConfMatrix_NBC, classLabels);
ConfMatrix_NBC.RowSummary = 'row-normalized';
title('NBC');

h(6) = figure;
ConfMatrix_KNN = confusionchart(ConfMatrix_KNN, classLabels);
ConfMatrix_KNN.RowSummary = 'row-normalized';
title('KNN');

h(7) = figure;
ConfMatrix_DA = confusionchart(ConfMatrix_DA, classLabels);
ConfMatrix_DA.RowSummary = 'row-normalized';
title('DA');

% SAVE RESULTS

savefig(h, 'C:\TFG\Matlab_Files\ResultPlots2.fig');

clear all;
close all;
clc;

%% ENTRENAMIENTO DE LOS MODELOS CON LOS DATOS CONJUNTOS OBTENIDOS EL 26/07/2021 Y EL 03/08/2021

% Network parameters

Network_parameters;

load('datos_gestos_combinados.mat');

GenerateDataset;

% DEEP LEARNING

        % LSTM + CNN

layersLSTMCNN = [...
    sequenceInputLayer(inputSize, 'Name', 'input')
    sequenceFoldingLayer('Name', 'fold')
    convolution2dLayer(filterSize, numFilters, 'Name', 'conv')
    batchNormalizationLayer('Name','bn')
    reluLayer('Name','relu')
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayer(numHiddenUnits,'OutputMode','last','Name','lstm')
    dropoutLayer(dropoutProbability, 'Name', 'doutprob')
    fullyConnectedLayer(numClasses, 'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

lgraphLSTMCNN = layerGraph(layersLSTMCNN);
lgraphLSTMCNN = connectLayers(lgraphLSTMCNN,'fold/miniBatchSize','unfold/miniBatchSize');

optionsLSTMCNN = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience',5, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',true, ...
    'VerboseFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.1);

accLSTMCNN = NaN(numIter, 1);
for i = 1:numIter
    [BestNetLSTMCNN(i), info] = trainNetwork(XTrain,YTrain,lgraphLSTMCNN,optionsLSTMCNN);
    accLSTMCNN(i) = info.FinalValidationAccuracy;
end
[BestAcc(1),n] = max(accLSTMCNN);
BestAcc(1) = BestAcc(1)/100;
netLSTMCNN = BestNetLSTMCNN(n);
YPredLSTMCNN = classify(netLSTMCNN, XValid, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTMCNN = confusionmat(YValid, YPredLSTMCNN);
clear BestNetLSTMCNN YPredLSTMCNN;

        % Long Short-Term Memory

optionsLSTM = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience',5, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',true, ...
    'VerboseFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.1);

layersLSTM = [ ...
    sequenceInputLayer(inputSize)
    flattenLayer
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(dropoutProbability)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

accLSTM = NaN(numIter, 1);
for i = 1:numIter
    [BestNetLSTM(i), info] = trainNetwork(XTrain,YTrain, layersLSTM, optionsLSTM);
    accLSTM(i) = info.FinalValidationAccuracy;
end
[BestAcc(2),n] = max(accLSTM);
BestAcc(2) = BestAcc(2)/100;
netLSTM = BestNetLSTM(n);
YPredLSTM = classify(netLSTM, XValid, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTM = confusionmat(YValid, YPredLSTM);
clear BestNetLSTM YPredLSTM;

        % Gated Recurrent Unit

layersGRU = [ ...
    sequenceInputLayer(inputSize)
    flattenLayer
    gruLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(dropoutProbability)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

optionsGRU = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'MaxEpochs',maxEpochs, ...
    'ValidationPatience',5, ...
    'ValidationData', {XValid, YValid}, ...
    'ValidationFrequency', 20, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',true, ...
    'VerboseFrequency', 100, ...
    'Shuffle', 'every-epoch', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 3, ...
    'LearnRateDropFactor', 0.1);

accGRU = NaN(numIter, 1);
for i = 1:numIter
    [BestNetGRU(i), info] = trainNetwork(XTrain,YTrain, layersGRU, optionsGRU);
    accGRU(i) = info.FinalValidationAccuracy;
end
[BestAcc(3),n] = max(accGRU);
BestAcc(3) = BestAcc(3)/100;
netGRU = BestNetGRU(n);
YPredGRU = classify(netGRU, XValid, 'MiniBatchSize', miniBatchSize);
ConfMatrix_GRU = confusionmat(YValid, YPredGRU);
clear BestNetGRU YPredGRU;


    % MACHINE LEARNING

accSVM = NaN(numIter, 1);
for j = 1:numIter
    [BestSVM(j), accSVM(j)] = trainSVM(Trainset);
end
[BestAcc(4), n] = max(accSVM);
SVM = BestSVM(n);
SVM_Pred = SVM.predictFcn(Validset);
ConfMatrix_SVM = confusionmat(Validset.Label, SVM_Pred);
clear BestSVM SVM_Pred;

accDA = NaN(numIter, 1);
for j = 1:numIter
    [BestDA(j), accDA(j)] = trainDA(Trainset);
end
[BestAcc(5), n] = max(accDA);
DA = BestDA(n);
DA_Pred = DA.predictFcn(Validset);
ConfMatrix_DA = confusionmat(Validset.Label, DA_Pred);
clear BestDA DA_Pred;

accKNN = NaN(numIter, 1);
for j = 1:numIter
    [BestKNN(j), accKNN(j)] = trainKNN(Trainset);
end
[BestAcc(6), n] = max(accKNN);
KNN = BestKNN(n);
KNN_Pred = KNN.predictFcn(Validset);
ConfMatrix_KNN = confusionmat(Validset.Label, KNN_Pred);
clear BestKNN KNN_Pred;

accNBC = NaN(numIter, 1);
for j = 1:numIter
    [BestNBC(j), accNBC(j)] = trainNBC(Trainset);
end
[BestAcc(7), n] = max(accNBC);
NBC = BestNBC(n);
NBC_Pred = NBC.predictFcn(Validset);
ConfMatrix_NBC = confusionmat(Validset.Label, NBC_Pred);
clear BestNBC NBC_Pred;

acc = [accLSTMCNN, accLSTM, accGRU, accSVM, accNBC, accKNN, accDA];
for i = 1:3
   acc(:,i) = acc(:,i)./100; 
end
clear accLSTMCNN accLSTM accGRU accSVM accNBC accKNN accDA;

clear n j;

% PLOT

h(1) = figure;
ConfMatrix_LSTMCNN = confusionchart(ConfMatrix_LSTMCNN, classLabels);
ConfMatrix_LSTMCNN.RowSummary = 'row-normalized';
title('CNN+LSTM');

h(2) = figure;
ConfMatrix_LSTM = confusionchart(ConfMatrix_LSTM, classLabels);
ConfMatrix_LSTM.RowSummary = 'row-normalized';
title('LSTM');

h(3) = figure;
ConfMatrix_GRU = confusionchart(ConfMatrix_GRU, classLabels);
ConfMatrix_GRU.RowSummary = 'row-normalized';
title('GRU');

h(4) = figure;
ConfMatrix_SVM = confusionchart(ConfMatrix_SVM, classLabels);
ConfMatrix_SVM.RowSummary = 'row-normalized';
title('SVM');

h(5) = figure;
ConfMatrix_NBC = confusionchart(ConfMatrix_NBC, classLabels);
ConfMatrix_NBC.RowSummary = 'row-normalized';
title('NBC');

h(6) = figure;
ConfMatrix_KNN = confusionchart(ConfMatrix_KNN, classLabels);
ConfMatrix_KNN.RowSummary = 'row-normalized';
title('KNN');

h(7) = figure;
ConfMatrix_DA = confusionchart(ConfMatrix_DA, classLabels);
ConfMatrix_DA.RowSummary = 'row-normalized';
title('DA');

h(8) = figure;
boxplot(acc);

% SAVE RESULTS

save C:\TFG\Matlab_Files\Resultados3.mat acc BestAcc DA KNN NBC SVM netGRU netLSTM netLSTMCNN;

savefig(h, 'C:\TFG\Matlab_Files\ResultPlots3.fig');

clear all;
close all;