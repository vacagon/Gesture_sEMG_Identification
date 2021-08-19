figure;
%% Puño cerrado
%Filtrado
subplot(8,4,[1,5]);
plot(Filtered_PC.TimeValues,Filtered_PC.DataValues);
title('Puño cerrado');
xlim([0, max(Filtered_PC.TimeValues)]);
ylim([-600, 600]);
%Normalizado
subplot(8,4,[9,13]);
plot(Normalized_PC.TimeValues,Normalized_PC.DataValues);
xlim([0, max(Filtered_PC.TimeValues)]);
ylim([0, 1.5]);
%Características
[MAV, WL, ZC, SSC, t] = feature_offline(Filtered_PC.DataValues, Normalized_PC.DataValues);
subplot(8,4,17);
plot(t, MAV);
title('MAV');
xlim([0, max(Filtered_PC.TimeValues)]);
ylim([0, 0.4]);
subplot(8,4,21);
plot(t, WL);
title('WL');
xlim([0, max(Filtered_PC.TimeValues)]);
ylim([-0.5, 0.5]);
subplot(8,4,25);
plot(t, ZC);
title('ZC');
xlim([0, max(Filtered_PC.TimeValues)]);
ylim([0, 80]);
subplot(8,4,29);
plot(t, SSC);
title('SSC');
xlim([0, max(Filtered_PC.TimeValues)]);
ylim([0, 130]);
%% Mano relajada
%Filtrado
subplot(8,4,[2,6]);
plot(Filtered_MR.TimeValues,Filtered_MR.DataValues);
title('Mano relajada');
xlim([0, max(Filtered_MR.TimeValues)]);
ylim([-600, 600]);
%Normalizado
subplot(8,4,[10,14]);
plot(Normalized_MR.TimeValues,Normalized_MR.DataValues);
xlim([0, max(Filtered_MR.TimeValues)]);
ylim([0, 1.5]);
%Características
[MAV, WL, ZC, SSC, t] = feature_offline(Filtered_MR.DataValues, Normalized_MR.DataValues);
subplot(8,4,18);
plot(t, MAV);
title('MAV');
xlim([0, max(Filtered_MR.TimeValues)]);
ylim([0, 0.4]);
subplot(8,4,22);
plot(t, WL);
title('WL');
xlim([0, max(Filtered_MR.TimeValues)]);
ylim([-0.5, 0.5]);
subplot(8,4,26);
plot(t, ZC);
title('ZC');
xlim([0, max(Filtered_MR.TimeValues)]);
ylim([0, 80]);
subplot(8,4,30);
plot(t, SSC);
title('SSC');
xlim([0, max(Filtered_MR.TimeValues)]);
ylim([0, 130]);
%% Muñeca flexionada
%Filtrado
subplot(8,4,[3,7]);
plot(Filtered_MF.TimeValues,Filtered_MF.DataValues);
xlim([0, max(Filtered_MF.TimeValues)]);
title('Muñeca flexionada');
ylim([-600, 600]);
%Normalizado
subplot(8,4,[11,15]);
plot(Normalized_MF.TimeValues,Normalized_MF.DataValues);
xlim([0, max(Filtered_MF.TimeValues)]);
ylim([0, 1.5]);
%Características
[MAV, WL, ZC, SSC, t] = feature_offline(Filtered_MF.DataValues, Normalized_MF.DataValues);
subplot(8,4,19);
plot(t, MAV);
title('MAV');
xlim([0, max(Filtered_MF.TimeValues)]);
ylim([0, 0.4]);
subplot(8,4,23);
plot(t, WL);
title('WL');
xlim([0, max(Filtered_MF.TimeValues)]);
ylim([-0.5, 0.5]);
subplot(8,4,27);
plot(t, ZC);
title('ZC');
xlim([0, max(Filtered_MF.TimeValues)]);
ylim([0, 80]);
subplot(8,4,31);
plot(t, SSC);
title('SSC');
xlim([0, max(Filtered_MF.TimeValues)]);
ylim([0, 130]);
%% Mano abierta
%Filtrado
subplot(8,4,[4,8]);
plot(Filtered_MA.TimeValues,Filtered_MA.DataValues);
title('Mano abierta');
xlim([0, max(Filtered_MA.TimeValues)]);
ylim([-600, 600]);
%Normalizado
subplot(8,4,[12,16]);
plot(Normalized_MA.TimeValues,Normalized_MA.DataValues);
xlim([0, max(Filtered_MA.TimeValues)]);
ylim([0, 1.5]);
%Características
[MAV, WL, ZC, SSC, t] = feature_offline(Filtered_MA.DataValues, Normalized_MA.DataValues);
subplot(8,4,20);
plot(t, MAV);
title('MAV');
xlim([0, max(Filtered_MA.TimeValues)]);
ylim([0, 0.4]);
subplot(8,4,24);
plot(t, WL);
title('WL');
xlim([0, max(Filtered_MA.TimeValues)]);
ylim([-0.5, 0.5]);
subplot(8,4,28);
plot(t, ZC);
title('ZC');
xlim([0, max(Filtered_MA.TimeValues)]);
ylim([0, 80]);
subplot(8,4,32);
plot(t, SSC);
title('SSC');
xlim([0, max(Filtered_MA.TimeValues)]);
ylim([0, 130]);