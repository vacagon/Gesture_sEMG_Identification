function [emg_signal_normal, emg_max] = Normalize(emg, clock, max)
%% Output
%emg_signal_normal: Valor de la señal de EMG normalizado
%emg_max: Valor máximo de amplitud de la EMG en t-1
%% Input
%emg: Señal de EMG de entrada (Rectificada)
%clock: Tiempo de simulación
%max: Valor máximo de amplitud de la EMG en t
%% Ignorar los primeros 2 segundos de señal
if clock <= 2
    emg_signal_normal = 0;
    emg_max = 0;
    
else
    %% Coger el valor máximo de amplitud durante los 18 segundos restantes
    if clock <= 20
        emg_signal_normal = 0;
        if emg > max
            emg_max = emg;
        else
            emg_max = max;
        end
        %% Normalizar el valor de la señal de entrada
    else
        emg_signal_normal = emg/max;
        %% Tomar el valor máximo en t como el proximo valor máximo en t-1 y realimentar con el la función
        emg_max = max;
    end
end

