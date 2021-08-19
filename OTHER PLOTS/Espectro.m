%% Se carga el archivo con los datos de simulación
load('Data_2005(2).mat');
Signal = Rectified4;
%% Frecuencia de muestreo 1 kHz
Fs = 1000;

%% Longitud de la señal
L = length(Signal.DataValues);

%% TRANSFORMADA RÁPIDA DE FOURIER
Y = fft(Signal.DataValues);

%% Espectro bilateral P2
P2 = abs(Y/L);

%% Espectro unilateral P1
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

%% Se define el dominio de la frecuencia f
f = Fs*(0:(L/2))/L;

%% Se representa el espectro de la señal
figure;
subplot(1,2,1);
plot(f,P1);
title('Espectro en frecuencia');
xlabel('f [Hz]');
%xlim([0,500]);
%ylim([0,1]);
subplot(1,2,2);
plot(Signal.TimeValues, Signal.DataValues);
xlim([20,max(Signal.TimeValues)]);
title('Señal de EMG');
xlabel('Tiempo [s]');
ylabel('Lectura del A/D');