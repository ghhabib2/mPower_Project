function [F1, F2, F3, F4, Voice, Pitch, Gender, Filter_Cutoff, Log_ratio_threshold1,...
    Log_ratio_threshold2, Autocorrelation_Threshold_Level] = mb_ftracker(X,Fs)

% FORMANT_TRACKER
% 
%  [F1, F2, F3, F4, Voice, Pitch, Gender, Filter_Cutoff,
%  Log_ratio_threshold1,
%  Log_ratio_threshold2,
%  Autocorrelation_Threshold_Level] = mb_ftracker(X,Fs);
%
%  This function estimates the first four formant frequencies of a speech
%  signal, X and plots the results on the narrowband spectrogram of X. X
%  has to be passed to the function in vector format. 
%  The sampling frequency of the speech signal has to be specified in Hz,
%  as Fs. 
%
%  The function returns the first four formant frequencies: F1, F2, F3,
%  and F4 (in Hz).
%  
%
% INPUTS
% X     The speech signal (vector)
% Fs    The sampling frequency of X (in Hz)
%
% OUTPUTS
% F1    First formant frequency estimates (in Hz)
% F2    Second formant frequency estimates (in Hz)
% F3    Third formant frequency estimates (in Hz)
% F4    Fourth formant frequency estimates (in Hz)
% Voice The vocing decisions for the signal (0 = Unvoiced and 1 = Voiced)
%
% NON-STANDARD FUNCTION CALLS
% formant_tracker_backend.m
%
% SEE ALSO
% WAVREAD, and AUREAD.
%
% TECHNICAL REFERENCES
%
% Primary reference:
%
% - K. Mustafa and I. C. Bruce, "Robust formant tracking for continuous
%   speech with speaker
%   variability," IEEE Transactions on Speech and Audio Processing, Mar. 2006.
%
% Additional reference:
%
% - K. Mustafa, "Robust formant tracking for continuous speech with speaker variability," 
%   M.A.Sc. dissertation, Dept. Elect. and Comp. Eng. McMaster Univ., Hamilton, ON, Canada, 2003.
%
%
% Authors: Kamran Mustafa  AND  Ian C. Bruce
% E-mail: mkamran@hotmail.com  OR  ibruce@ieee.org
%
% (c) 2004-2006

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% General Parameters Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Downsample signal to 8 KHz
X = resample(X,8e3,Fs);

%New Sampling Frequency (after downsampling)
Fs = 8e3;

%LPC Window size assignment - 20 ms
lpc_window_size = 0.02*Fs;

%Declare the number of formant filters to create
num_of_filters = 4;

%Create Hamming Window - of duration lpc_window_size (20 ms)
window = repmat(hamming(lpc_window_size,'periodic'),1,num_of_filters);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% VOicing Detector Parameters Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initial cut-off frequency for the voicing detector HPF and LPF (in Hz)
Filter_Cutoff_init = 700;

%The Hysterisis Log Ratio Threshold1 for switching from unvoiced speech to voiced speech (0 --> 1)
Log_ratio_threshold1_init = 0.2;

%The Hysterisis Log Ratio Threshold2 for switching from voiced speech to unvoiced speech (1 --> 0)
Log_ratio_threshold2_init = 0.3; 

%The Threshold Level for use with the Autocorrelation vs. logratio calculation for white noise consideration 
Autocorrelation_Threshold_Level_init = 0.4;

%Set Initial RMS Ratio threshold values for each formant
RMS_Ratio_F1 = -35;
RMS_Ratio_F2 = -40;
RMS_Ratio_F3 = -45;
RMS_Ratio_F4 = -50;


%Call the Formant Tracking function
[F, Voice, Pitch, Gender, Filter_Cutoff, Log_ratio_threshold1, Log_ratio_threshold2, Autocorrelation_Threshold_Level] = formant_tracker_backend (X, Fs, lpc_window_size, window, Filter_Cutoff_init, Log_ratio_threshold1_init, Log_ratio_threshold2_init, Autocorrelation_Threshold_Level_init, RMS_Ratio_F1, RMS_Ratio_F2, RMS_Ratio_F3, RMS_Ratio_F4);

F1 = F(1,:);
F2 = F(2,:);
F3 = F(3,:);
F4 = F(4,:);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLOT THE RESULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%figure;
%specgram(X,512,Fs,256,round(0.925*256))
%[low_r high_r] = caxis;
%caxis ([(high_r-60) high_r]);
%xlabel('\bf Time (s)');
%ylabel('\bf Frequency (Hz)');
%title('\bf Spectrogram, estimated formant frequencies and voicing detection');
%hold on;
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),F1,'k','linewidth',3.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),F2,'k -.','linewidth',3.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),F3,'k --','linewidth',3.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),F4,'k :','linewidth',3.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),250*Voice,'m --','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),Pitch,'g --','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),50*Gender,'r --','linewidth',2.5)
%legend ('\bf F1', '\bf F2', '\bf F3', '\bf F4', '\bf Voicing', '\bf Pitch', '\bf Gender')

%figure;
%specgram(X,512,Fs,256,round(0.925*256))
%[low_r high_r] = caxis;
%caxis ([(high_r-60) high_r]);
%xlabel('\bf Time (s)');
%ylabel('\bf Frequency (Hz)');
%title('\bf Spectrogram, voicing, pitch, gender, and various parameter values');
%hold on;
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),250*Voice,'m --','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),Pitch,'g --','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),50*Gender,'r --','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),Filter_Cutoff,'k-','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),1e4*Log_ratio_threshold1,'k--','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),1e4*Log_ratio_threshold2,'k-.','linewidth',2.5)
%plot([0:1/Fs:(length(X)-1)/Fs]-(((lpc_window_size+256)/2)+10)*(1/Fs),5e3*Autocorrelation_Threshold_Level,'-','color',[0.5 0.5 0.5],'linewidth',2.5)
%legend ('\bf Voicing', '\bf Pitch', '\bf Gender', '\bf F_{vd}', '\bf LT_v', '\bf -LT_u', '\bf \alpha')
