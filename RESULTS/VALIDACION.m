clear variables;
clc;

load('RESULTS.mat');
load('Valid_data.mat');

miniBatchSize = 30;
classLabels = ["Mano Relajada" "Puño Cerrado" ...
    "Muñeca Flexionada" "Mano Abierta" "Muñeca Extendida"];

%% CNN + LSTM

YPredLSTMCNN = classify(netLSTMCNN, X, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTMCNN = confusionmat(Y, YPredLSTMCNN);

%% LSTM

YPredLSTM = classify(netLSTM, X, 'MiniBatchSize', miniBatchSize);
ConfMatrix_LSTM = confusionmat(Y, YPredLSTM);

%% GRU

YPredGRU = classify(netGRU, X, 'MiniBatchSize', miniBatchSize);
ConfMatrix_GRU = confusionmat(Y, YPredGRU);

%% SVM

SVM_Pred = SVM.predictFcn(Dataset);
ConfMatrix_SVM = confusionmat(Dataset.Label, SVM_Pred);

%% NBC

NBC_Pred = NBC.predictFcn(Dataset);
ConfMatrix_NBC = confusionmat(Dataset.Label, NBC_Pred);

%% KNN

KNN_Pred = KNN.predictFcn(Dataset);
ConfMatrix_KNN = confusionmat(Dataset.Label, KNN_Pred);

%% DA

DA_Pred = DA.predictFcn(Dataset);
ConfMatrix_DA = confusionmat(Dataset.Label, DA_Pred);

%% PLOTS

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

%% SAVE RESULTS

savefig(h, 'C:\TFG\Matlab_Files\RESULTS\ValidPlots.fig');

clear all;
close all;
clc;