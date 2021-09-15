Fs = 10000;
duration_secs = 2;
t = [0:1/Fs:duration_secs];

% Specs of the carrier affect specs of the modulating signal
amp_carrier = 20;
freq_carrier = 4500;

% Build our modulating signal m
target_modulation_index = 1;
freq_m = 5 * sin(2*pi*t) + 10;

amp_m = sin(2*pi*t) + 1;
m = (amp_m) .* sin(2*pi*t .* freq_m);
m = m + sin(2*pi*10*t);

% I'm slightly unclear on the modulation index, but it has to do with our
% modulating signal peak amplitude compared to our carrier signal peak
% amplitude. Basically, if our modulating signal is too strong we wash out
% (m>1 -> overmodulated). Under modulation is less bad, but we still want a
% modulation index of 1.
modulation_index = max(m) / amp_carrier;
m = m ./ (modulation_index/target_modulation_index);

% https://www.tutorialspoint.com/analog_communication/analog_communication_am_modulators.htm
% This shows it as a summation...

carrier = amp_carrier .* sin(2 * pi * freq_carrier * t);
x = (amp_carrier + m) .* sin(2 * pi * freq_carrier * t);


%%%%%%%%%%%%%%
% DEMODULATE %
%%%%%%%%%%%%%%
rectified = x;
rectified( rectified <= 0) = 0;

filtered = lowpass(rectified, 5, Fs);

tiledlayout(5,1);
nexttile;
plot(m);

nexttile;
plot(carrier);

nexttile;
plot(x);

nexttile;
plot(rectified);

nexttile;
plot(filtered);