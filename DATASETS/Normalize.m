function [emg_signal_normal, emg_max] = Normalize(emg, clock, max)
%% Output
%emg_signal_normal: Valor de la se�al de EMG normalizado
%emg_max: Valor m�ximo de amplitud de la EMG en t-1
%% Input
%emg: Se�al de EMG de entrada (Rectificada)
%clock: Tiempo de simulaci�n
%max: Valor m�ximo de amplitud de la EMG en t
%% Ignorar los primeros 2 segundos de se�al
if clock <= 2
    emg_signal_normal = 0;
    emg_max = 0;
    
else
    %% Coger el valor m�ximo de amplitud durante los 18 segundos restantes
    if clock <= 20
        emg_signal_normal = 0;
        if emg > max
            emg_max = emg;
        else
            emg_max = max;
        end
        %% Normalizar el valor de la se�al de entrada
    else
        emg_signal_normal = emg/max;
        %% Tomar el valor m�ximo en t como el proximo valor m�ximo en t-1 y realimentar con el la funci�n
        emg_max = max;
    end
end

