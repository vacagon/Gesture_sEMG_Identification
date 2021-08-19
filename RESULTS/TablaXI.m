load('RESULTS_valid.mat');

rowLabels = [ "Media" "Mediana" "Mejor" "Peor" ];
columnLabels = [ "CNN + LSTM" "LSTM" "GRU" "SVM" "NBC" "KNN" "DA" ];

CNNLSTM = [mean(acc(:,1)); median(acc(:,1)); max(acc(:,1)); min(acc(:,1))];
LSTM = [mean(acc(:,2)); median(acc(:,2)); max(acc(:,2)); min(acc(:,2))];
GRU = [mean(acc(:,3)); median(acc(:,3)); max(acc(:,3)); min(acc(:,3))];
SVM = [mean(acc(:,4)); median(acc(:,4)); max(acc(:,4)); min(acc(:,4))];
NBC = [mean(acc(:,5)); median(acc(:,5)); max(acc(:,5)); min(acc(:,5))];
KNN = [mean(acc(:,6)); median(acc(:,6)); max(acc(:,6)); min(acc(:,6))];
DA = [mean(acc(:,7)); median(acc(:,7)); max(acc(:,7)); min(acc(:,7))];

A = [CNNLSTM, LSTM, GRU, SVM, NBC, KNN, DA];

T = array2table(A, ...
    'VariableNames', columnLabels, ...
    'RowNames', rowLabels );