Fs = 10000;
duration_secs = 2;
t = [0:1/Fs:duration_secs];

% Build the AM signal
amp_carrier = 20;
freq_carrier = 4500;
target_modulation_index = 1;

freq_m = 200; % Results in sidebands at 4300 and 4700!
amp_m = 2;

m = (amp_m) .* sin(2*pi*t .* freq_m);
modulation_index = max(m) / amp_carrier;
m = m ./ (modulation_index/target_modulation_index);
x = (amp_carrier + m) .* sin(2 * pi * freq_carrier * t);

tiledlayout(3,1);

nexttile;
plot(m);

nexttile;
plot(x);

L = numel(t);
Y = fft(x);
P2 = abs(Y);
P2 = P2/L;
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

% The frequency axis. We have L/2 bins, and Fs/2 Hz spread across them. So
% that is (Fs/2)/(L/2) Hz per Bin = Fs/L Hz Per Bin
f = Fs/L*(0:(L/2));

nexttile;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')