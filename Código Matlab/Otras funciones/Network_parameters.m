segmentLength = 200;
numChannels = 3;
inputSize = [1, segmentLength, numChannels];
numHiddenUnits = 100;
numClasses = 5;
maxEpochs = 30;
miniBatchSize = 30;
dropoutProbability = 0.1;
filterSize = 1;
numFilters = 20;
numIter = 50;
BestAcc = NaN(1, 7);
acc = NaN(numIter, 7);
classLabels = ["Mano Relajada" "Puño Cerrado" ...
    "Muñeca Flexionada" "Mano Abierta" "Muñeca Extendida"];