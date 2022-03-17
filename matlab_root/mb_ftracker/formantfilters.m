function [BT, AT] = formantfilters(Fs, Pitch, formant_frequencies)

% FORMANTFILTERS
% 
%  [BT, AT] = formantfilters(Fs, Pitch, formant_frequencies)
%  
%  This function calculates the filter coefficients for the four formant filters and returns the 
%  filter coefficients in vectors BT (numerator) and AT (denominator).
%  
%
% INPUTS
% Fs                    The sampling frequency of the original signal (in Hz)
% Pitch                 The pitch frequency (in Hz)
% formant_frequencies   The four formant frequency values (in Hz)
%
% OUTPUTS
% BT                    Filter coefficients of the numerator
% Voice                 Filter coefficients of the denominator
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
% Additional reference:
%
% - K. Mustafa, "Robust formant tracking for continuous speech with speaker variability," 
%   M.A.Sc. dissertation, Dept. Elect. and Comp. Eng. McMaster Univ., Hamilton, ON, Canada, 2003.

%
% Authors: Kamran Mustafa  AND  Ian C. Bruce
% E-mail: mkamran@hotmail.com  OR  ibruce@ieee.org
%
% (c) 2004-2006

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize the return variables
AT = zeros(4,2);
BT = zeros(4,5);

%Setup the general DTF parameters
Rp = 0.9;
K = 1-Rp; %The DC gain

%set-up the PITCH DTF pole at the location of the Pitch
Fk0 = Pitch;
P0 = Rp*exp(j*2*pi*Fk0/Fs); %Location of the pole - Pitch dependent

%set-up the 1st DTF pole at the first Formant Frequency location
Fk1 = formant_frequencies(1);
P1 = Rp*exp(j*2*pi*Fk1/Fs); %Location of the pole

%set-up the 2nd DTF pole at the second Formant Frequency location
Fk2 = formant_frequencies(2);
P2 = Rp*exp(j*2*pi*Fk2/Fs); %Location of the pole

%set-up the 3rd DTF pole at the third Formant Frequency location
Fk3 = formant_frequencies(3);
P3 = Rp*exp(j*2*pi*Fk3/Fs); %Location of the pole

%set-up the 4th DTF pole at the fourth Formant Frequency location
Fk4 = formant_frequencies(4);
P4 = Rp*exp(j*2*pi*Fk4/Fs); %Location of the pole


%set-up the 4 AZF Parameters
Rz = .98;
Fl0 = Pitch;
Fl1 = formant_frequencies(1);
Fl2 = formant_frequencies(2);
Fl3 = formant_frequencies(3);
Fl4 = formant_frequencies(4);

%set-up the Pitch dependent AZF
Z0 = [Rz*exp(j*2*pi*Fl0/Fs)]; %The single zero at the pitch location 

%set-up the first AZF
Z1 = [Rz*exp(j*2*pi*Fl1/Fs)]; %The single zero location

%set-up the second AZF  
Z2 = [Rz*exp(j*2*pi*Fl2/Fs)]; %The single zero location

%set-up the third AZF  
Z3 = [Rz*exp(j*2*pi*Fl3/Fs)]; %The single zero location

%set-up the fourth AZF 
Z4 = [Rz*exp(j*2*pi*Fl4/Fs)]; %The single zero location


%FIRST TRACKING FILTER COEFFICIENTS
%Construct the first formant tracking filter with a pole at the 1st formant frequency estimate
%setup the appropriate all zero filter DC gain values
Kn0 = 1/(1-Rz*exp(j*2*pi*((Fl0-Fk1)/Fs))); %The zero filter DC gain - for the pitch
Kn2 = 1/(1-Rz*exp(j*2*pi*((Fl2-Fk1)/Fs))); %The zero filter DC gain
Kn3 = 1/(1-Rz*exp(j*2*pi*((Fl3-Fk1)/Fs))); %The zero filter DC gain
Kn4 = 1/(1-Rz*exp(j*2*pi*((Fl4-Fk1)/Fs))); %The zero filter DC gain
%The overall Filter Function
PT1 = P1; %pole location
ZT1 = [Z0; Z2; Z3; Z4]; %combine the zero locations
KT1 = K.*Kn0.*Kn2.*Kn3.*Kn4; %combine the DC gain terms of the AZF and the DTF
%Obtain the filter coefficients for the 1st filter
[BT(1,:), AT(1,:)]  = zp2tf_complex (ZT1,PT1,KT1); %Final filter values at this pole location


%SECOND TRACKING FILTER COEFFICIENTS
%Construct the second formant tracking filter with a pole at the 2nd formant frequency estimate
%setup the appropriate all zero filter DC gain values
Kn1 = 1/(1-Rz*exp(j*2*pi*((Fl1-Fk2)/Fs))); %The zero filter DC gain
Kn3 = 1/(1-Rz*exp(j*2*pi*((Fl3-Fk2)/Fs))); %The zero filter DC gain
Kn4 = 1/(1-Rz*exp(j*2*pi*((Fl4-Fk2)/Fs))); %The zero filter DC gain
%The overall Filter Function
PT2 = P2; %pole location
ZT2 = [Z1; Z3; Z4]; %combine the zero locations
KT2 = K.*Kn1.*Kn3.*Kn4; %combine the DC gain terms of the AZF and the DTF
%Obtain the filter coefficients for the 2nd filter
[BT(2,1:4), AT(2,:)]  = zp2tf_complex (ZT2,PT2,KT2); %Final filter values at this pole location


%THIRD TRACKING FILTER COEFFICIENTS
%Construct the third formant tracking filter with a pole at the 3rd formant frequency estimate
%setup the appropriate all zero filter DC gain values
Kn1 = 1/(1-Rz*exp(j*2*pi*((Fl1-Fk3)/Fs))); %The zero filter DC gain
Kn2 = 1/(1-Rz*exp(j*2*pi*((Fl2-Fk3)/Fs))); %The zero filter DC gain
Kn4 = 1/(1-Rz*exp(j*2*pi*((Fl4-Fk3)/Fs))); %The zero filter DC gain
%The overall Filter Function
PT3 = P3; %pole location
ZT3 = [Z1; Z2; Z4]; %combine the zero locations
KT3 = K.*Kn1.*Kn2.*Kn4; %combine the DC gain terms of the AZF and the DTF
%Obtain the filter coefficients for the 3rd filter
[BT(3,1:4), AT(3,:)]  = zp2tf_complex (ZT3,PT3,KT3); %Final filter values at this pole location


%FOURTH TRACKING FILTER COEFFICIENTS
%Construct the fourth formant tracking filter with a pole at the 4th formant frequency estimate
%setup the appropriate all zero filter DC gain values
Kn1 = 1/(1-Rz*exp(j*2*pi*((Fl1-Fk4)/Fs))); %The zero filter DC gain
Kn2 = 1/(1-Rz*exp(j*2*pi*((Fl2-Fk4)/Fs))); %The zero filter DC gain
Kn3 = 1/(1-Rz*exp(j*2*pi*((Fl3-Fk4)/Fs))); %The zero filter DC gain
%The overall Filter Function
PT4 = P4; %pole location
ZT4 = [Z1; Z2; Z3]; %combine the zero locations
KT4 = K.*Kn1.*Kn2.*Kn3; %combine the DC gain terms of the AZF and the DTF
%Obtain the filter coefficients for the 4th filter
[BT(4,1:4), AT(4,:)]  = zp2tf_complex (ZT4,PT4,KT4); %Final filter values at this pole location
