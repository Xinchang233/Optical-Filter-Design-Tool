function [re, rd, u, D, ripple] = para_conv(S, M, B, N)

% output:
%        re -- ring-bus coupling rate 1;
%        rd -- ring-bus coupling rate 2;
%         u -- ring-ring coupling rate;
% % %          These parameters are normalized by ro
%        ripple -- the ratio between peak transmission and resonance
%                  transmission;
%         D -- peak transmission

% input
%         S, M
%         B -- bandwidth normalized by 2ro
%         N -- N-dB for bandwidth

bandwidth = B;    %Δω_NdB/2ro
if S<=1
    ripple = 1;
else
    ripple = (S+1)^2/(4*S);
end

N = N - 10/log(10)*log(ripple);

a = 10^(N/10);
re = (1+M)*bandwidth/(sqrt(S-1+sqrt(a*S^2+2*(a-2)*S+a)))-1;
rd = (1-M)*bandwidth/(sqrt(S-1+sqrt(a*S^2+2*(a-2)*S+a)))-1;
u = 0.5*sqrt(S*(re+rd+2)^2+(re-rd)^2);

if isreal(re) && re>0 && isreal(rd) && rd>0 && isreal(u) && u>0 && ripple<N
    tag = 1;
else
    tag = 0;
end

if tag == 0;
    D = nan;
    warning('Choosen parameters yeild an unphyscial result.')
else
    D = 4*u^2*re*rd/(u^2+(re+1)*(rd+1))^2*ripple;
end



end