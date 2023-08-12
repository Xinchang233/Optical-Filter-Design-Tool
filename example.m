clc,clear;close all


% This is an example showing how to use para_conv.m

% please contact zhangxc@bu.edu if you have any question

color1 = [0.639215686274510	0.207843137254902	0.168627450980392];

r0 = 1; % arbitrary value


S = 0.2333; % Define passband shape
M = 0; % Define impedance mismatch
N = 10; % N value in "fixed N-dB bandwidth"
Bandwidth = 10; % Full or two-sided N-dB bandwidth
B = Bandwidth/(2*r0); % Full or two-sided N-dB bandwidth normalized by 2r0

[re, rd, u, D] = para_conv(S, M, B, N); % Get coupling rates and peak transmission

dw = -2*B:0.01:2*B; % Define frequency detuing for plot

transmission = 4*u^2*re*rd./(dw.^4+dw.^2.*((re+r0)^2+(rd+r0)^2-2*u^2)+(u^2+(re+r0)*(rd+r0))^2); % Drop port transmission using Eq. (S4)
tran_dB = 10/log(10)*log(transmission); % Convert into dB (power)
 
figure; 
subplot(1,2,2) % Transmission and its N dB bandwidth
plot(dw,tran_dB,"Color",color1,'LineWidth',2)
hold on
grid on
set(gca,'Color','none','fontsize',16)
grid on
xlabel('Frequency detuning','fontsize',16)
ylabel('Dorp port transmission |\its_d \rm/\its_i\rm|^2 (dB)','fontsize',16)
x = [-0.5*Bandwidth 0.5*Bandwidth]; % Find frequency detuning for N dB decay
y = 10/log(10)*log(D) - N + 0.*x; 
plot(x,y,"Color",'k','LineWidth',2) % Plot horizontal line to illustrate N dB bandiwdth
txt = [num2str(N),' dB bandiwdth'];
text(0,y(1)+0.5,txt,'FontSize',14,'BackgroundColor','none')


Y = (re+2*r0+rd)/2; X = sqrt(u^2-(re-rd)^2/4);

if isreal(X) % get the real and imaginary part of two poles
    Y1 = Y;
    Y2 = Y;
    X1 = X;
    X2 = -X;
else
    Y1 = Y + imag(X);
    Y2 = Y - imag(X);
    X1 = 0; X2 = 0;
end


subplot(1,2,1)
plot([X1 X2],[Y1 Y2],'x','markersize',10,'linewidth',2,'Color',color1)

        hold on
        
        plot([0 X1],[0 Y1],'--','LineWidth',1,'Color',color1);plot([0 X2],[0 Y2],'--','LineWidth',1,'Color',color1)
        axis equal; grid off;
        set(gca,'LineWidth',1.5,'XAxisLocation','origin','YAxisLocation','origin','Box','off','xtick',[],'ytick',[])
        xlim([-abs(X1)/0.618 abs(X1)/0.618]); ylim([0 abs(Y1)*2])
        xlabel('\itδω\rm_r','FontSize',16)
        ylabel('\itδω\rm_i','FontSize',16)
        title('Pole configuration (to scale)','FontSize',16)
        set(gca,'color','none')











