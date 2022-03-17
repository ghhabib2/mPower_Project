function [gender, avgF0] = gender_detector(X,Fs)

% GENDER_DETECTOR
% 
%  [gender, avgF0] = gender_detector(X,Fs)
%  
%  This function calculates gender of the speaker and returns the pitch frequency and
%  the gender of the primary speaker.
%  
%
% INPUTS
% X                     The speech signal (vector)
% Fs                    The sampling frequency of X (in Hz)
% Pitch                 The pitch frequency (in Hz)
% formant_frequencies   The four formant frequency values (in Hz)
%
% OUTPUTS
% gender                The gender of the primary speaker (0 = Male and 1 = Female)
% avgF0                 The pitch frequency (in Hz)
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
% - L.R. Rabiner, and R.W. Schafer, Digital processing of speech signals. Englewood Cliffs, NJ, Prentice Hall, 1978.

% Authors: Kamran Mustafa  AND  Ian C. Bruce
% E-mail: mkamran@hotmail.com  OR  ibruce@ieee.org
%
% (c) 2004-2006

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Turn Warnings off
warning off;

%Figure out the number of samples in X
n_samples = length(X);

% Window update rate - Window spacing - 5 ms
updRate=floor(5*Fs/1000);  

%Window size - 20 ms
fRate=floor(20*Fs/1000); %Use a 20 ms window

%Number of frames in this sample
nFrames=floor((n_samples-fRate)/updRate)+1; 

%Initialize variables
avgF0=0;
f01=zeros(1,nFrames);
k=1;
m=1;


%Calculate over all the frames in the sample 
for t=1:nFrames
    
        
    %Select the window (20 ms) over which to do the pitch estimation
    X_Win= X(k:k+fRate-1);       
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do the Pitch Estimation over one frame %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Remove the DC bias 
    X_Win = X_Win - mean(X_Win);
    
    %LPF the speech at 900 Hz to remove upper freq. (since Pitch info. will be below 900 Hz)
    [bf0,af0]=butter(4,900/(Fs/2));
    X_Win = filter(bf0,af0,X_Win); 
    
    %Perform Centre Clipping and find the Clipping Level (CL)
    i13=fRate/3;
    maxi1=max(abs(X_Win(1:i13)));
    
    i23=2*fRate/3;
    maxi2=max(abs(X_Win(i23:fRate)));
    
    %Choose the appropriate clipping level
    if maxi1>maxi2    
        CL=0.68*maxi2;       
    else 
        CL= 0.68*maxi1;
    end %endif maxi1>maxi2  
    
    %Perform Center clipping   
    clip=zeros(fRate,1);
    ind1=find(X_Win>=CL);
    clip(ind1)=X_Win(ind1)-CL;
    
    ind2=find(X_Win <= -CL);
    clip(ind2)=X_Win(ind2)+CL;
    
    engy=norm(clip,2)^2;
    
    %Compute the autocorrelation 
    RR=xcorr(clip);
    g=fRate;
    
    
    %The pitch estimates are limited to being between 60 and 320 Hz
    %Find the max autocorrelation in the range 60 Hz <= F0 <= 320 Hz
    LF=floor(Fs/320); 
    HF=floor(Fs/60);
    
    Rxx=abs(RR(g+LF:g+HF));
    % From Here
    [rmax, imax]= max(Rxx);
    imax=imax+LF;
    
    %Estimate raw pitch
    pitch=Fs/imax;

    %Check max RR against V/UV threshold
    silence=0.4*engy;
    
    %Make Voiced/Unvoiced descions
    if (rmax > silence)  & (pitch > 60) & (pitch <=320)
        pitch=Fs/imax;
    else
        % It is a unvoiced segment   
        pitch=0;
    end %endif (rmax > silence)  & (pitch > 60) & (pitch <=320)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of Pitch Estimation per Window %%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Assign Pitch estimate
    f0(t)=pitch;

    % Do median filtering
    if t>2 & nFrames>3
        
        z=f0(t-2:t); 
        md=median(z);
        f01(t-2)=md;
        
        if md > 0
            avgF0=avgF0+md;
            m=m+1;
        end %endif md > 0
        
    elseif nFrames<=3
        
        disp('# of frames less than 3');
        f01(t)=pitch;
        avgF0=avgF0+pitch;
        m=m+1;
        
    end %endif t>2 & nFrames>3
    
    %Put next window 'updRate' appart from where the last one was.
    k=k+updRate;
    
end %endfor t=1:nFrames


%Calculate the avg. pitch estimate for the whole sample X.
if m==1
    avgF0=0;
else
    avgF0=avgF0/(m-1);
end %endif m==1



%Find Gender (The pitch being used is the avg. pitch - avgF0)
if (avgF0 >= 180)

    %It's female
    gender = 1;
    
else
    
    %It's male
    gender = 0;
    
end %endif 
    
%Turn Warnings back on
warning on;