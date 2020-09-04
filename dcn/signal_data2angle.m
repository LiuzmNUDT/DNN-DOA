%% 产生数据
clc
clear variables
close all
M=8;snapshot=256;
f0=1e6;
fc=1e6;
fs=4*f0;
C=M*(M-1);
%%
MaxItr = 800;
ErrorThr = 1e-3;     % RVM 终止误差
D_start=-60;
D_stop=59;
LM=1000;
theta=D_start:1:D_stop;
L=length(theta);
A=exp(1i*pi*fc*(0:M-1)'*sind(theta)/f0);
H=zeros(M*M,L);
for i=1:M
    fhi=A*diag(exp(-1i*pi*(i-1)*sind(theta)));
    H((i-1)*M+1:i*M,:)=fhi;
end
Angle=2:1:15;
S_label=zeros(LM,L,length(Angle));
gamma=zeros(LM,L,length(Angle));
gamma_R=zeros(LM,L,length(Angle));
R_est=zeros(LM,C,length(Angle));
S_est=zeros(LM,L,2,length(Angle));
DOA_train=zeros(2,LM,length(Angle));
S_abs=zeros(LM,L,length(Angle));
T_SBC=zeros(LM,length(Angle));
T_SBC_R=zeros(LM,length(Angle));

for k=1:length(Angle)
    for i=1:LM
        DOA_train(1,i,k)=-Angle(k)/2+1*rand-0.5;
        DOA_train(2,i,k)=DOA_train(1,i,k)+Angle(k);
        [X1,~]=signal_generate(M,snapshot,DOA_train(1,i,k),f0,fc,fs,1);
        [X2,~]=signal_generate(M,snapshot,DOA_train(2,i,k),f0,fc,fs,1);
        temp1=awgn(X1,0,'measured');
        temp2=awgn(X2,0,'measured');
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


save('data2_angle.mat','R_est','DOA_train','gamma_R',...
    'S_label','S_est','theta','gamma','T_SBC','T_SBC_R','Angle')
%data2_angle1  一个角度固定为0
%data2_angle  法线对称
figure
 plot(mean(T_SBC))
 hold on
plot(mean(T_SBC_R))
