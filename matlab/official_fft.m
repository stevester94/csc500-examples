Fs = 500;            % Sampling frequency                    
T = 1/Fs;             % Sampling period
duration_secs = 10;
t = (0:T:duration_secs);        % Time vector
L = numel(t);

S = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t) + 5;
X = S;

figure(1);
plot(1000*t(1:50),X(1:50))
title('Original Signal')
xlabel('t (milliseconds)')
ylabel('X(t)')

Y = fft(X);

% The ouput of FFT are complex sinusoids. What exactly this means I am not
% sure. But get the magnitude of each point using abs();
P2 = abs(Y);

% The output is scaled by the length of the input signal, so we have to
% normalize it.
P2 = P2/L;

% The fft is mirrored across its midpoint, so we drop the second half
P1 = P2(1:L/2+1);

% Double it, because reasons.
%
P1(2:end-1) = 2*P1(2:end-1);

% The frequency axis. We have L/2 bins, and Fs/2 Hz spread across them. So
% that is (Fs/2)/(L/2) Hz per Bin = Fs/L Hz Per Bin
f = Fs/L*(0:(L/2));
figure(2);
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')