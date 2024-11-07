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

% Tworzenie podziału 80% na treningowe i 20% na testowe
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

% Rzutowanie rzeczywistych etykiet oraz przewidywanych do odpowiedniego formatu
yTestCategorical = categorical(yTest);
yPredClassCategorical = categorical(yPredClass);

% Wyświetlenie macierzy pomyłek
figure;
plotconfusion(yTestCategorical, yPredClassCategorical);
title('Validation Confusion Matrix');
