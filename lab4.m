clear all
clc

% Dla wierszy 7 i 8

dane = load('Data_PTC_vs_FTC.mat');

x = dane.Data.X;
d = dane.Data.D;

X = [x(7,:); x(8,:)]';

zlozenie = [x(7,:);x(8,:);d];


% Własna metoda 
% 
% net = feedforwardnet(10);
% 
% net = train(net,[x(7,1:43);x(8,1:43)],d(1:43));
% 
% sim(net)
% 

% Tworzenie podziału 
c = cvpartition(size(X, 1), 'HoldOut', 0.5);

% Indeksy dla zbioru treningowego i testowego
XTrain = X(training(c), :);
yTrain = d(training(c));
XTest = X(test(c), :);
yTest = d(test(c));


% Tworzenie sieci neuronowej z 10 neuronami w warstwie ukrytej
net = feedforwardnet(10);

% Trenowanie sieci na zbiorze treningowym
net = train(net, XTrain', yTrain); 

% Symulacja na zbiorze testowym
yPred = net(XTest'); % Używamy wytrenowanej sieci do przewidywania wyników


% Konwersja wyników z sieci (ciągłe wartości) na klasy binarne (opcjonalnie, jeśli potrzebne)
yPredClass = round(yPred); % Zaokrąglenie wyników do 0 lub 1, jeśli to klasyfikacja binarna
yPredClass(yPredClass == 0) = -1;

Ytest = sim(net,XTest');
plot(XTest,Ytest,'o')


figure;
gscatter(XTest(:, 1), XTest(:, 2), yPredClass, 'rb', 'xo');
xlabel('Feature 1');
ylabel('Feature 2');
title('SVM Predictions');
legend('Class 1', 'Class -1');


% Rzutowanie rzeczywistych etykiet oraz przewidywanych do odpowiedniego formatu
yTestCategorical = categorical(yTest);
yPredClassCategorical = categorical(yPredClass);

% Wyświetlenie macierzy pomyłek
figure;
plotconfusion(yTestCategorical, yPredClassCategorical);
title('Validation Confusion Matrix');

accuracy = sum(yPredClass == yTest') / length(yTest);


%% SVM
% Tworzenie modelu SVM
SVMModel = fitcsvm(XTrain, yTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Predykcja na zbiorze testowym
yPredSVM = predict(SVMModel, XTest);

% Konwersja wyników na klasy binarne (opcjonalnie, jeśli w danych masz -1 i 1)
yPredClassSVM = yPredSVM;
yPredClassSVM(yPredClassSVM == 0) = -1;

% Rzutowanie rzeczywistych etykiet oraz przewidywanych do odpowiedniego formatu
yTestCategoricalSVM = categorical(yTest);
yPredClassCategoricalSVM = categorical(yPredClassSVM);

% Wyświetlenie macierzy pomyłek
figure;
plotconfusion(yTestCategoricalSVM, yPredClassCategoricalSVM');
title('Validation Confusion Matrix');

% Opcjonalnie: wizualizacja wyników
figure;
gscatter(XTest(:, 1), XTest(:, 2), yPredClassSVM, 'rb', 'xo');
xlabel('Feature 1');
ylabel('Feature 2');
title('SVM Predictions');
legend('Class 1', 'Class -1');

% Obliczanie punktów ROC i AUC
[~, scores] = predict(SVMModel, XTest); % 'scores' to margines SVM
[rocX, rocY, ~, AUC] = perfcurve(yTest, scores(:, 2), 1);

accuracySVM = sum(yPredClassSVM == yTest') / length(yTest);

% Wyrysowanie krzywej ROC
figure;
plot(rocX, rocY, 'b', 'LineWidth', 2);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(['ROC Curve (AUC = ' num2str(AUC) ')']);
grid on;