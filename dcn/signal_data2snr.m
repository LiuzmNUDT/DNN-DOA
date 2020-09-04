%% 产生数据
clc
clear variables
close all
M=8;snapshot=256;
f0=1e6;
fc=1e6;
fs=4*f0;
times=1000;
MaxItr = 800;
ErrorThr = 1e-3;     % RVM 终止误差

D_start=-60;
D_stop=59;
K=2;
theta=D_start:1:D_stop;
L=length(theta);
A=exp(1i*pi*fc*(0:M-1)'*sind(theta)/f0);
H=zeros(M*M,L);
for i=1:M
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*M+1:i*M,:)=fhi;
end
SNR=-15:1:15;
S_label=zeros(times,L,length(SNR));
R_est=zeros(times,M*(M-1),length(SNR));
gamma=zeros(times,L,length(SNR));
gamma_R=zeros(times,L,length(SNR));
T_SBC=zeros(times,length(SNR));
T_SBC_R=zeros(times,length(SNR));
S_est=zeros(times,L,2,length(SNR));
DOA_train=zeros(2,times,length(SNR));

for k=1:length(SNR)
    for i=1:times
        DOA_train(1,i,k)=-10+1*rand-0.5;
        DOA_train(2,i,k)=DOA_train(1,i,k)+15;
        
        [X1,~]=signal_generate(M,snapshot,DOA_train(1,i,k),f0,fc,fs,1);
        [X2,~]=signal_generate(M,snapshot,DOA_train(2,i,k),f0,fc,fs,1);
        temp1=awgn(X1,SNR(k),'measured');
        temp2=awgn(X2,SNR(k),'measured');
        X= temp1+ temp2;
        [R_est(i,:,k),Rx]=feature_extract_R(X) ;
        temp=H'*vec(Rx);
        temp=temp/norm(temp);
        S_est(i,:,1,k)=real(temp);
        S_est(i,:,2,k)=imag(temp);
        
        
        S_label(i,round(DOA_train(1,i,k))+61,k)=1;
        S_label(i,round(DOA_train(2,i,k))+61,k)=1;
        tic
        [gamma_R(i,:,k)] = SBAC_R(X,H,MaxItr,ErrorThr, S_label(i,:),2);
        T_SBC_R(i,k)=toc;
        tic
        [gamma(i,:,k)] = SBAC(X,A,MaxItr,ErrorThr);
        T_SBC(i,k)=toc;
        
        i
        k
    end
   
end

close all
plot(theta,S_est(i,:,1,k))
xlim([-60,60])
hold on
plot(theta,(S_label(i,:,k)'))
grid on
plot(theta,(gamma(i,:,k)'))
plot(theta,(gamma_R(i,:,k)'))
legend('A','true','S','R')
save('data2_snr.mat','R_est','DOA_train',...
    'S_label','S_est','theta','SNR','gamma','gamma_R','T_SBC','T_SBC_R')


