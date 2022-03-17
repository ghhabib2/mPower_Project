function Y = firhilbert (X)

% FIRHILBERT
% 
%  Y = firhilbert (X)
%  
%  This function calculates analytic version of the real valued signal X using an FIR Hilbert Transformer.
%  
%
% INPUTS
% X     The real valued speech signal (vector)
%
% OUTPUTS
% Y     The complex valued representation of the speech signal (vector)
%
% NON-STANDARD FUNCTION CALLS
% zp2tf_complex.m
% 
% TECHNICAL REFERENCES
%
% Primary reference:
%
% - K. Mustafa and I. C. Bruce, "Robust formant tracking for continuous speech with speaker
%   variability," IEEE Transactions on Speech and Audio Processing, Mar. 2006.
%
% Additional references:
%
% - K. Mustafa, "Robust formant tracking for continuous speech with speaker variability," 
%   M.A.Sc. dissertation, Dept. Elect. and Comp. Eng. McMaster Univ., Hamilton, ON, Canada, 2003.
% - L.R. Rabiner, and R.W. Schafer, "On the behavior of minimax FIR digital Hilbert transformers,"
%   The Bell System Technical Journal, Vol. 53, No. 2, 1974.
% - J. Picone, D. P. Prezas, W. T. Hartwell, and J. L. Locicero, 
%   "Spectrum estimation using an analytic signal representation," Signal Processing, vol. 15, no. 2, pp. 169-182, 1988.

% Authors: Kamran Mustafa  AND  Ian C. Bruce
% E-mail: mkamran@hotmail.com  OR  ibruce@ieee.org
%
% (c) 2004-2006

% Define Filter parameters
Filter_Length = 20;

%Remez Filter Parameters
B_Remez = remez(Filter_Length,[.1 .9],[1 1],'Hilbert');
A_Remez = 1;

%Filtered Values - Immaginary part of the Signal
X_Im = filter(B_Remez,A_Remez,X);

%Delay Filter Parameters 
B_Delay = [zeros(1,10),1,zeros(1,10)];
A_Delay = 1;

%Delayed - Real part of the Signal
X_Re = filter(B_Delay,A_Delay,X);

% Summ up the Real and the Immaginary parts
Y = X_Re + 1j*X_Im;
