function Features = feature_offline(filtered, normalized)
l = length(filtered);
filtered = buffer(filtered, 200, 50, 'nodelay');
normalized = buffer(normalized, 200, 50, 'nodelay');
[~,n] = size(filtered);
signal_mav = zeros(1,n);
signal_wl = zeros(1,n);
signal_zc = zeros(1,n);
signal_ssc = zeros(1,n);
for i = 1:n
   signal_mav(1,i) = MAV(normalized(:,i));
   signal_wl(1,i) = WL(normalized(:,i));
   signal_zc(1,i) = ZC(filtered(:,i));
   signal_ssc(1,i) = SSC(filtered(:,i));
end
MeanAbsoluteValue = signal_mav;
WaveformLength = signal_wl;
ZeroCrossing = signal_zc;
SlopeSignChanges = signal_ssc;
Features = [MeanAbsoluteValue; WaveformLength; ZeroCrossing; SlopeSignChanges];
end