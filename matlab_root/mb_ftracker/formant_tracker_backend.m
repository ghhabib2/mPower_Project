function [F, Voice, Pitch, Gender, Filter_Cutoff, Log_ratio_threshold1, Log_ratio_threshold2,...
    Autocorrelation_Threshold_Level]...
    = formant_tracker_backend(X, Fs, lpc_window_size, window, Filter_Cutoff_init,...
    Log_ratio_threshold1_init, Log_ratio_threshold2_init, Autocorrelation_Threshold_Level_init,...
    RMS_Ratio_F1, RMS_Ratio_F2, RMS_Ratio_F3, RMS_Ratio_F4)

% FORMANT_TRACKER_BACKEND
% 
% [F, Voice, Pitch, Gender, Filter_Cutoff, Log_ratio_threshold1, ...
% Log_ratio_threshold2, Autocorrelation_Threshold_Level] = formant_tracker_backend(X, Fs, lpc_window_size, window, Filter_Cutoff_init, Log_ratio_threshold1_init, Log_ratio_threshold2_init, Autocorrelation_Threshold_Level_init, RMS_Ratio_F1, RMS_Ratio_F2, RMS_Ratio_F3, RMS_Ratio_F4);
%
%  This function returns the first four formant frequencies of the signal, X. It performs all the calculations
%  outlined in the technical references to estimate the formants.
%  
%
% INPUTS
% X                                 The speech signal vector
% Fs                                The sampling frequency of X (in Hz)
% lpc_window_size                   The size of the window over which LPC is performed (in # of samples)
% window                            The coefficients of the windowing function applied to the signal
% Filter_Cutoff                     The cutoff value of the HPF and LPF of the voicing detector (in Hz)
% Log_ratio_threshold1              The Hysterisis threshold value for switching from unvoiced to voiced speech (0 --> 1)
% Log_ratio_threshold2              The Hysterisis threshold value for switching from voiced to unvoiced speech (1 --> 0)
% Autocorrelation_Threshold_Level   The threshold level for the autocorrelation test
% RMS_Ratio_F1 ... RMS_Ratio_F4     The initial energy threshold values for each formant frequency
%
% OUTPUTS
% F                                 The matrix containing the formant frequency estimates (in Hz), each row is a formant
% Voice                             The vocing decisions for the signal (0 = Unvoiced and 1 = Voiced)
%
% NON-STANDARD FUNCTION CALLS
% formantfilters.m, gender_detector.m, firhilbert.m, rms.m, db.m
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


%Turn off warnings
warning off;

%Equalize the input file to have an RMS of 1
X = X./rms(X);

%Replicate the origianl signal to be sent to the voicing detector
X_VD = X';

%Apply the Pre-Amphasis filter to the input signal
Xhf_preemphasis = filter([0.2618 -0.2618],[1 0.4764],X); %Butterworth IIR filter, Order 1, stable 

%Hilbert Transform the signal into a complex signal using the FIR Hilbert function.
X = firhilbert(Xhf_preemphasis);
X=X.';

%Initialize pitch and formant frequencies

%For normal operation, use the same initial values for F0 - F1 every time
%that the algorithm is run.
%If you wish to replicate the random initial values used in the Mustafa and
%Bruce (2006) paper, change the line "if (1)" below to "if (0)".

if (1) 

    %Fixed initial values for F0 - F1, set at the mean values from Table II
    %of Mustafa and Bruce (2006).
    
    F0to4_init = [150; 458; 1535; 2705; 3819];

else

    %Random initial values for F0 - F1, with means and variances given in
    %Table II of Mustafa and Bruce (2006).
    
    F0to4_init = zeros(5,1);

    while (any(F0to4_init <=0)|any(F0to4_init >= Fs/2)|any(diff(F0to4_init)<=0))

        %Generate 5 random numbers from a zero-mean, unity-variance Gaussian
        %distribution

        Q = randn(size(F0to4_init));

        %Assign initial pitch and formant frequency values to the pitch and each
        %of the formants based on experimental values (randn*std + mean)

        F0to4_init = [Q(1)*40+150; Q(2)*154+458; Q(3)*469+1535; Q(4)*468+2705; Q(5)*485+3819];

    end

end

Pitch_Initial = F0to4_init(1);
F1 = F0to4_init(2);
F2 = F0to4_init(3);
F3 = F0to4_init(4);
F4 = F0to4_init(5);

%Intialize F_freq
F_freq = [F1;F2;F3;F4];

%Initial Filter assignments
[B,A] = formantfilters(Fs, Pitch_Initial, [F_freq(1), F_freq(2), F_freq(3), F_freq(4)]);
B = B.';
A = [A.';zeros(3,4)]; %To equalize A into a 5 x 5 matrix as well (like B).

%Initialize the Avg. Formant Frequency
Avg_F_freq = [F1; F2; F3; F4];

%Initialize the Moving Average of the RMS Levels 
Avg_Y_RMS = [RMS_Ratio_F1; RMS_Ratio_F2; RMS_Ratio_F3; RMS_Ratio_F4];
Avg_Y_RMS = repmat(Avg_Y_RMS,1,lpc_window_size+1);

%Set the previous formant frequency
Last_F_freq = F_freq; 

%Order of the filter - doesn't change
[tmp, num_of_filters] = size(B);
filter_order =tmp-1;

%Zero Pad the input signal to deal with the initialization
Y = zeros(num_of_filters,size(X,2));
X_VD = [(zeros(1,filter_order)) X_VD];
X = [(zeros(1,filter_order)) X];
Y_temp = zeros(num_of_filters,size(X,2));

%Frequency movement indicators (they are only here to keep track of the movement)- initialize
F1_Mov(1:lpc_window_size+1) = F1;
F2_Mov(1:lpc_window_size+1) = F2;
F3_Mov(1:lpc_window_size+1) = F3;
F4_Mov(1:lpc_window_size+1) = F4;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%VOICING DETECTOR and GENDER DETECTOR INITIALIZATIONS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Initialize the return variable: 1 for voiced speech; and 0 for unvoiced speech
Voice = zeros(1,length(X_VD));

%Initialize the gender 0 for male and 1 for female and -1 for no change from last (hold at previous)
Gender = zeros(1,length(X_VD));
Pitch = zeros(1,length(X_VD));


%Initialize the voicing detector parameters
Filter_Cutoff = zeros(1,length(X_VD));
Log_ratio_threshold1 = zeros(1,length(X_VD));
Log_ratio_threshold2 = zeros(1,length(X_VD));
Autocorrelation_Threshold_Level = zeros(1,length(X_VD));

Filter_Cutoff(1:filter_order) = Filter_Cutoff_init;
Log_ratio_threshold1(1:filter_order) = Log_ratio_threshold1_init;
Log_ratio_threshold2(1:filter_order) = Log_ratio_threshold2_init;
Autocorrelation_Threshold_Level(1:filter_order) = Autocorrelation_Threshold_Level_init;

%Initialize the Avg. Pitch
Avg_Pitch = Pitch_Initial;

%Setup Jump counter to an initial value of 50 ms - Determines how often the Pitch/Gender is checked 
Jump_Counter = Fs/20;

%The HPF Parameters - 20th order high-pass Butterworth filter.
[B_HPF,A_HPF] = butter(20,Filter_Cutoff_init/(Fs/2), 'high');

%The LPF Parameters - 20th order low-pass Butterworth filter
[B_LPF,A_LPF] = butter(20,Filter_Cutoff_init/(Fs/2));

X_HPF = zeros(1,length(X_VD));
X_LPF = zeros(1,length(X_VD));
Zi_HPF = [];
Zi_LPF = [];

%Delay Filter Parameters 
B_Delay = [zeros(1,10),1,zeros(1,10)];
A_Delay = 1;

%Delayed part - for Autocorrelation
X_Delayed= filter(B_Delay,A_Delay,X_VD);

%Setup the waitbar.
wb1 = waitbar(0,'Running...');
set(wb1,'name','Formant Tracker - 0%');

%Compute over entire signal
for n=filter_order+1:length(X)
    
    %Compute filtered signal for each filter (each formant)
    for c = 1:num_of_filters
        
        Y_temp(c,n) = [X(n:-1:(n-filter_order))]*[B(1:(filter_order+1),c)] - [Y_temp(c,n-1:-1:(n-filter_order))]*[A(2:(filter_order+1),c)];
        
    end %inner loop endfor c = 1:num_of_filters
    
    %Calculate the RMS values (complex) of the filetred signals
    Y_RMS(:,n) = rms(Y_temp(:,max(n-(lpc_window_size-1),1):n).').';
    
    %%%%%%%%%%%%%%%%%%%%%%% VOICING DETECTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %High-pass part - for log ratio calculation - Voicing Detector
    [X_HPF(n),Zi_HPF] = filter(B_HPF,A_HPF,X_VD(n),Zi_HPF);

    %Low-pass part - for log ration calculation - Voicing Detector
    [X_LPF(n),Zi_LPF] = filter(B_LPF,A_LPF,X_VD(n),Zi_LPF);
    
    %Calculate autocorrelation on a moving square window of the same size as the LPC window - 20 ms
    if (n > lpc_window_size+1) 
        %autocorrelation of the delayed signal on a moving 20 ms square window - after LPC kicks in
        autocorr = xcorr(X_Delayed(n-(lpc_window_size-1):n));
    else
        %set the autocorrelation to zero for the length of the signal before the LPC kicks in.
        autocorr = zeros(2*lpc_window_size-1,1);
    end %endif (n > lpc_window_size+1) 
    
    %Calculate the logratio of the LPF and the HPF 
    Log_ratio(n) = log((rms(X_LPF(max(1, n-(lpc_window_size-1)):n))/sqrt(Filter_Cutoff(n-1)))/(rms(X_HPF(max(1, n-(lpc_window_size-1)):n))/(sqrt(Fs/2-Filter_Cutoff(n-1)))));
    
    %- HYSTERESIS - Assign voiced/unvoiced to each of the data points in the sample
    %If the previous sample was unvoiced AND the current sample had a log ratio of MORE than the 
    %log ratio threshold1 (for switcing from 0 --> 1), then the current sample is voiced. ie. 
    %The switch from unvoiced to voiced occurs only if the log ratio threshold 1 is crossed.
    if ((Voice(max(1,n-1)) == 0) & (Log_ratio(n) > Log_ratio_threshold1(n-1)))
        
        %set current sample to be voiced speech
        Voice(n) = 1;
        
        %If the previous sample was voiced AND the current sample had a log ratio of LESS than the 
        %log ratio threshold2 (for switcing from 1 --> 0), then the current sample is unvoiced. ie. 
        %The switch from voiced to unvoiced occurs only if the log ratio threshold 2 is crossed.
    elseif ((Voice(max(1,n-1)) == 1) & (Log_ratio(n) < -(Log_ratio_threshold2(n-1))))
        
        %set current sample to be unvoiced speech
        Voice(n) = 0;
        
        %If the log ratio threshold is NOT crossed then assign the current sample to be like the last one    
    else
        Voice(n) = Voice(max(1,n-1));
        
    end %endif Hysterisis.
    
    %Use the results for the autocorrelator AND the hysteresis to make a final decisions regarding voiced/unvoiced speech.
    %The sample is voiced if the logratio says it voiced AND the the autocorrelation at at least one point in the window
    %is greater than 0.25 (Autocorrelation_Level) times the autocorrelation at the centre of the window AND
    %there is at least one point in the window whose autocorrelation is greater than 0.
    
    Voice(n) = (Voice(n) & any(abs(autocorr([1:lpc_window_size-1 lpc_window_size+1:2*lpc_window_size-1])) >= Autocorrelation_Threshold_Level(n-1)*autocorr(lpc_window_size)) & any(abs(autocorr) > 0));
    
    %%%%%%%%%%%%%%%%%%%%%%% GENDER DETECTOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Satisfy indexing and length requirements for the Gender Detector
    %Check Gender iff sample is voiced AND the check hasn't been done in a while (determined by Jump_Counter)
    if ( (Voice(n) == 1) & (n >= Jump_Counter) )
        
        %Setup Jump counter to setup how often the pitch is going to be checked - every 20 ms
        Jump_Counter = n + (Fs/50);
        
        %Setup the windowed data to send to the Gender Detector
        X_GD = X_VD((n-399):n);
        
        %Call the Gender Detector and obtain pitch as well
        [Gender(n), Pitch(n)] = gender_detector(X_GD, Fs);
        
        %Round the Pitch to the closest integer
        Pitch(n) = round(Pitch(n));
        
    else
        Gender(n) = Gender(n-1);
        Pitch(n) = Pitch(n-1);
    end %endif ( (Voice(n) == 1) & (n >= Jump_Counter) )  
    
    
    %Move Voicing Detector threshold levels depending on the result from the Gender Detector - Males
    if (Gender(n) == 0)
        
        %It's MALE AND we need to change ANY one of the parameters any further
        if (Filter_Cutoff(n-1) > 700) | (Log_ratio_threshold1(n-1) > 0.1) | (Log_ratio_threshold2(n-1) > 0.2) | (Autocorrelation_Threshold_Level(n-1) < 0.6)    
            
            %Change filter_cutoff gradually, if requied
            if (Filter_Cutoff(n-1) > 700)
                
                %Decay fast enough so that it can change from one end to the other in ~44 ms
                Filter_Cutoff(n) = Filter_Cutoff(n-1) - 1.2;
                
                %If there has been a change in the parameters re-calculate the LPF and the HPF for the Voicing Detector
                %The HPF Parameters - 20th order high-pass Butterworth filter.
                [B_HPF,A_HPF] = butter(20,Filter_Cutoff(n)/(Fs/2), 'high');
                
                %The LPF Parameters - 20th order low-pass Butterworth filter
                [B_LPF,A_LPF] = butter(20,Filter_Cutoff(n)/(Fs/2));
                                
            else 
                
                Filter_Cutoff(n) = Filter_Cutoff(n-1);
                
                
            end %endif (Filter_Cutoff > 700)
            
            %Change Log_ratio_threshold1 gradually, if requied
            if (Log_ratio_threshold1(n-1) > 0.1)
                
                %Decay fast enough so that it can change from one end to the other in 40 ms
                Log_ratio_threshold1(n) = Log_ratio_threshold1(n-1) - 3.125e-4;
                
            else 
                
                Log_ratio_threshold1(n) = Log_ratio_threshold1(n-1);
                
            end %endif (Log_ratio_threshold1 > 0.1)
            
            %Change Log_ratio_threshold2 gradually, if requied
            if (Log_ratio_threshold2(n-1) > 0.2)
                
                %Decay fast enough so that it can change from one end to the other in 40 ms
                Log_ratio_threshold2(n) = Log_ratio_threshold2(n-1) - 3.125e-4;
                
            else 
                
                Log_ratio_threshold2(n) = Log_ratio_threshold2(n-1);                  
                
            end %endif (Log_ratio_threshold2 > 0.2)
            
            %Change Autocorrelation_Threshold_Level gradually, if requied
            if (Autocorrelation_Threshold_Level(n-1) < 0.6) 
                
                %Decay fast enough so that it can change from one end to the other in ~44 ms
                Autocorrelation_Threshold_Level(n) = Autocorrelation_Threshold_Level(n-1) + 0.001;
                
            else 
                
                Autocorrelation_Threshold_Level(n) = Autocorrelation_Threshold_Level(n-1);                                                 
                
            end %endif (Autocorrelation_Threshold_Level < 0.6) 
            
        else 
            
            Filter_Cutoff(n) = Filter_Cutoff(n-1);
            Log_ratio_threshold1(n) = Log_ratio_threshold1(n-1);
            Log_ratio_threshold2(n) = Log_ratio_threshold2(n-1);
            Autocorrelation_Threshold_Level(n) = Autocorrelation_Threshold_Level(n-1);                              
            
        end %endif (Filter_Cutoff > 700) | (Log_ratio_threshold1 > 0.1) | (Log_ratio_threshold2 > 0.2) | (Autocorrelation_Threshold_Level < 0.6)
        
    elseif (Gender(n) == 1)%- Females
        
        %It's FEMALE AND we need to change ANY one of the parameters any further
        if (Filter_Cutoff(n-1) < 1120) | (Log_ratio_threshold1(n-1) < 0.2) | (Log_ratio_threshold2(n-1) < 0.3) | (Autocorrelation_Threshold_Level(n-1) > 0.25)    
            
            %Change filter_cutoff gradually, if requied
            if (Filter_Cutoff(n-1) < 1120)
                
                %Decay fast enough so that it can change from one end to the other in ~44 ms
                Filter_Cutoff(n) = Filter_Cutoff(n-1) + 1.2;
                
                %If there has been a change in the parameters re-calculate the LPF and the HPF for teh Voicing Detector                  
                %The HPF Parameters - 20th order high-pass Butterworth filter.
                [B_HPF,A_HPF] = butter(20,Filter_Cutoff(n)/(Fs/2), 'high');
                
                %The LPF Parameters - 20th order low-pass Butterworth filter
                [B_LPF,A_LPF] = butter(20,Filter_Cutoff(n)/(Fs/2));
                               
            else 
                
                Filter_Cutoff(n) = Filter_Cutoff(n-1);
                
            end %endif (Filter_Cutoff < 1120)
            
            %Change Log_ratio_threshold1 gradually, if requied
            if (Log_ratio_threshold1(n-1) < 0.2)
                
                %Decay fast enough so that it can change from one end to the other in 40 ms
                Log_ratio_threshold1(n) = Log_ratio_threshold1(n-1) + 3.125e-4;
                
            else 
                
                Log_ratio_threshold1(n) = Log_ratio_threshold1(n-1);
                
            end %endif (Log_ratio_threshold1 < 0.2)
            
            %Change Log_ratio_threshold2 gradually, if requied
            if (Log_ratio_threshold2(n-1) < 0.3)
                
                %Decay fast enough so that it can change from one end to the other in 40 ms
                Log_ratio_threshold2(n) = Log_ratio_threshold2(n-1) + 3.125e-4;
                
            else 
                
                Log_ratio_threshold2(n) = Log_ratio_threshold2(n-1);
                
            end %endif (Log_ratio_threshold2 < 0.3)
            
            %Change Autocorrelation_Threshold_Level gradually, if requied
            if (Autocorrelation_Threshold_Level(n-1) > 0.25) 
                
                %Decay fast enough so that it can change from one end to the other in ~44 ms
                Autocorrelation_Threshold_Level(n) = Autocorrelation_Threshold_Level(n-1) - 0.001;
                
            else 
                
                Autocorrelation_Threshold_Level(n) = Autocorrelation_Threshold_Level(n-1);                              
                
            end %endif (Autocorrelation_Threshold_Level > 0.25) 
            
        else 
            
            Filter_Cutoff(n) = Filter_Cutoff(n-1);
            Log_ratio_threshold1(n) = Log_ratio_threshold1(n-1);
            Log_ratio_threshold2(n) = Log_ratio_threshold2(n-1);
            Autocorrelation_Threshold_Level(n) = Autocorrelation_Threshold_Level(n-1);                                             
            
        end %endif (Filter_Cutoff < 1120) | (Log_ratio_threshold1 < 0.2) | (Log_ratio_threshold2 < 0.3) | (Autocorrelation_Threshold_Level > 0.25)    
        
    end %endif (Gender == 0)
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Pitch Calculations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Smooth out the pitch - avoid it from going to zero and jumping around during voiced speech
    if ( (Pitch(n) == 0) & (Voice(n) == 1) )
        
        %Hold Pitch at previous value
        Pitch(n) = Pitch(n-1);
        
    end %endif ( (Pitch(n) == 0) & (Voice(n) == 1) )
    
    %Set the Pitch to the moving average during unvoiced sections
    if ((Voice(n) == 0) )
        
        % Set Pitch to a moving avg. 
        Pitch(n) = Avg_Pitch;
        
    end %endif ( (Pitch(n) == 0) & (Voice(n) == 0) )
    
    %Update the moving average of the Pitch
    Avg_Pitch = ((((n-1) .* Avg_Pitch) + Pitch(n)) ./ n);
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Moving Average Calculations and LPC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Calcualte as long as the last sample has not been reached
    if ((n > lpc_window_size+1) & (n <= length(X)))
        
        %If the entire window is voiced
        if(all(Voice(n-(lpc_window_size-1):n)) == 1)
            
            %1st order LPC formant region calculations ON WINDOWED DATA
            F_lpc = lpc (window.*Y_temp(:,(n-lpc_window_size+1):n).',1);
            
            %convert LPC values into formant frequencies
            F_freq = sort(angle(-F_lpc(:,2))/(2*pi))*Fs;
            
            %Check to see if any of the formant results from the LPC are invalid frequencies
            if(isempty(find(isfinite(F_freq) ==0)) == 0)
                
                %Set all 'NaN' frequencies to the last valid Formant Frequency 
                F_freq(find(isfinite(F_freq) == 0)) = Last_F_freq;
                
            end %endif (isempty(find(isfinite(F_freq) ==0)) == 0)
            
            %Deal with any Negative Frequencies by Decaying to Average
            if (any(F_freq < 0))
                
                %Find the Formant frequencies that are less than ZERO
                F_freq_Bad = find(F_freq < 0);
                
                %Decay Each of the Negative Frequencies
                F_freq(F_freq_Bad) = (Last_F_freq(F_freq_Bad) - (0.002*(Last_F_freq(F_freq_Bad) - Avg_F_freq(F_freq_Bad))));
                
            end %endif
            
            %Update the RMS_Thresholds based on a moving average (within set limits) if the sentence is VOICED
            Avg_Y_RMS(:,n) = ((((n-1) .* Avg_Y_RMS(:,n-1)) + db(Y_RMS(:,n))) ./ n);
            
            
            %Check to see if any of the formant frequencies are below the Threshold Levels
            %If the Formant Frequency is less then the threshold level (moving threshold parameter - Avg_Y_RMS) then decay ONLY that formant
            if (db(Y_RMS(1,n)) < (Avg_Y_RMS(1,n) -6))
                
                F_freq(1) = Last_F_freq(1) - (0.002*(Last_F_freq(1) - Avg_F_freq(1)));
                
            end %endif (db(Y_RMS(1,n)) < Avg_Y_RMS(1,n))
            
            if ( (db(Y_RMS(2,n)) < (Avg_Y_RMS(2,n) -8)) | (abs(F_freq(2)-Last_F_freq(2)) > 900) ) 
                
                F_freq(2) = Last_F_freq(2) - (0.002*(Last_F_freq(2) - Avg_F_freq(2)));
                
            end %endif (db(Y_RMS(2,n)) < Avg_Y_RMS(1,n))
            
            if ( (db(Y_RMS(3,n)) < Avg_Y_RMS(3,n) - 10) | (abs(F_freq(3)-Last_F_freq(3)) > 900) )
                
                F_freq(3) = Last_F_freq(3) - (0.002*(Last_F_freq(3) - Avg_F_freq(3)));
                
            end %endif (db(Y_RMS(3,n)) < Avg_Y_RMS(3,n))
            
            if (db(Y_RMS(4,n)) < Avg_Y_RMS(4,n) - 14)
                
                F_freq(4) = Last_F_freq(4) - (0.002*(Last_F_freq(4) - Avg_F_freq(4)));
                
            end %endif (db(Y_RMS(4,n)) < Avg_Y_RMS(4,n))
            
            %Update the moving average Formant Frequency
            Avg_F_freq = ((((n-1) .* Avg_F_freq) + F_freq) ./ n);
            
        else %elseif(all(Voice(n-(lpc_window_size-1):n)) == 1) %If the entire window is NOT voiced
            
            %Decay the Formant Frequency for ALL formants
            F_freq = (Last_F_freq - (0.002*(Last_F_freq - Avg_F_freq)));
            
            %Decay the Avg_Y_RMS value based on the current RMS values and the Average RMS value.
            if (all(db(Y_RMS(:,n)) > (Avg_Y_RMS(:,n-1) - 5)))
                
                Avg_Y_RMS(:,n) = Avg_Y_RMS(:,n-1) - (0.002*(Avg_Y_RMS(:,n-1) - db(Y_RMS(:,n)))); 
                
            else
                
                Avg_Y_RMS(:,n) = Avg_Y_RMS(:,n-1);
                
            end %endif (all(db(Y_RMS(:,n)) > (Avg_Y_RMS(:,n-1) - 5)))
            
        end %endif (all(Voice(n-(lpc_window_size-1):n)) == 1)
        
        %Limit how close the formants are allowed to come to each other
        %F1 should not get closer than 150 Hz to the Pitch
        if (F_freq(1) < (Pitch(n)+150))
            
            F_freq(1) = Pitch(n) + 150;
            
        end %endif (F_freq(1) < (Pitch(n)+150))
        
        %F2 should not get closer than 300 Hz to F1
        if (F_freq(2) < (F_freq(1)+ 300))
            
            F_freq(2) = F_freq(1) + 300;
            
        end %endif (F_freq(2) < (F_freq(1)+300))
        
        %F3 should not get closer than 400 Hz to F2
        if (F_freq(3) < (F_freq(2)+ 400))
            
            F_freq(3) = F_freq(2) + 400;
            
        end %endif (F_freq(3) < (F_freq(2)+400))
        
        %F4 should not get closer than 500 Hz to F3, but should not be
        %greater than the Nyquist frequency
        if (F_freq(4) < (F_freq(3)+500))
            
            F_freq(4) = F_freq(3) + 500;
            
        end %endif (F_freq(4) < (F_freq(3)+500))

        %F4 should not exceed the Nyquist frequency
        F_freq(4) = min(F_freq(4),Fs/2-1);
        
        %Set the previous formant frequency to the final assignment for the current formant frequency
        Last_F_freq = F_freq;      
        
        %Re-Calculate Formant Filter Parameters
        [B,A] = formantfilters(Fs, Pitch(n), [F_freq(1), F_freq(2), F_freq(3), F_freq(4)]);
        B = B.';
        A = [A.';zeros(3,4)]; %To equalize A into a 4 x 4 matrix as well (like B).
        
        %re-assign Formant Frequency Tracking info. based on the Voicing Detector Info.
        F1 = F_freq(1);
        F2 = F_freq(2);
        F3 = F_freq(3);
        F4 = F_freq(4);
        
        %Frequency movement indicators - update
        F1_Mov(n) = F1;
        F2_Mov(n) = F2;
        F3_Mov(n) = F3;
        F4_Mov(n) = F4;
        
    end %end if (n > lpc_window_size+1) 
    
    %UPDATE WAIT-BAR every 1%
    if (mod(n,(round(length(X_VD)/100))) == 0)
        waitbar(n/length(X_VD),wb1)
        set(wb1,'name',['Formant Tracker - ' sprintf('%2.1f',n/length(X_VD)*100) '%'])
    end
    
end %end for - outter loop

%Close Waitbar
close(wb1)

%Turn warnings back on
warning on;

% Remove the extra padded info. from the matrix
Y = Y_temp(:,(filter_order+1):end);
X = X(:,(filter_order+1):end);
X_VD = X_VD(:,(filter_order+1):end);
Voice = Voice(:,(filter_order+1):end);
F1_Mov = F1_Mov(:,(filter_order+1):end);
F2_Mov = F2_Mov(:,(filter_order+1):end);
F3_Mov = F3_Mov(:,(filter_order+1):end);
F4_Mov = F4_Mov(:,(filter_order+1):end);
Gender = Gender(:,(filter_order+1):end);
Pitch = Pitch(:,(filter_order+1):end);
Filter_Cutoff = Filter_Cutoff(:,(filter_order+1):end);
Log_ratio_threshold1 = Log_ratio_threshold1(:,(filter_order+1):end);
Log_ratio_threshold2 = Log_ratio_threshold2(:,(filter_order+1):end);
Autocorrelation_Threshold_Level = Autocorrelation_Threshold_Level(:,(filter_order+1):end);

%Formant Frequencies
F = [F1_Mov; F2_Mov; F3_Mov; F4_Mov];