% This file will download required files and folders that my toolbox relies
% on to compute some of the dysphonia measures. The user may wish to take
% this step manually. The author of this toolbox has no responsibility for 
% suggesting the downloading of third party software

% Copyright (c) A. Tsanas, 2014

unzip('http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.zip', 'voicebox')
unzip('http://perso.ens-lyon.fr/patrick.flandrin/pack_emd.zip', 'EMD')
unzip('http://www.maxlittle.net/software/fastdfa.zip', 'DFA')
unzip('http://www.maxlittle.net/software/rpde.zip', 'RPDE')
unzip('http://www.mathworks.com/matlabcentral/fileexchange/submissions/1230/v/1/download/zip', 'SHRP'); % if there is any problem, search online for "shrp Matlab" and download the set of functions
urlwrite('http://www.cise.ufl.edu/~acamacho/publications/swipep.m', 'swipep.m')
