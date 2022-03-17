function y = db(x)
%dB	Decibels.
%	db(x) converts x to decibels.

y = 20*log10(abs(x));
