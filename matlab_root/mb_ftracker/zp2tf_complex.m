function [num,den] = zp2tf_complex(z,p,k)
%ZP2TF  Zero-pole to transfer function conversion.
%   [NUM,DEN] = ZP2TF(Z,P,K)  forms the transfer function:
%
%                   NUM(s)
%           H(s) = -------- 
%                   DEN(s)
%
%   given a set of zero locations in vector Z, a set of pole locations
%   in vector P, and a gain in scalar K.  Vectors NUM and DEN are 
%   returned with numerator and denominator coefficients in descending
%   powers of s.  
%
%   See also TF2ZP.

%   Note: modified from TF2ZP by Ian Bruce to work if p or z have elements
%   not in complex conjugate pairs.

den = poly(p(:)); % modified by I. Bruce to handle complex poles not in
                  % conjugate pairs
[md,nd] = size(den);
k = k(:);
[mk,nk] = size(k);
if isempty(z), num = [zeros(mk,nd-1),k]; return; end
[m,n] = size(z);
if mk ~= n
    if m == 1
        error('Z and P must be column vectors.');
    end
    error('K must have as many elements as Z has columns.');
end
for j=1:n
    zj = z(:,j);
    pj = poly(zj)*k(j); % modified by I. Bruce to handle complex zeros not
                        % in conjugate pairs
    num(j,:) = [zeros(1,nd-length(pj)) pj];
end
