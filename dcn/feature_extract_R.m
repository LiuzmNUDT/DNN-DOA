function [r_doa1,Rx]=feature_extract_R(X) 
%clc; clear all;X=randn(5,5);
[M,N]=size(X);
Rx=X*X'/N;%计算阵列协方差矩阵 时域协方差矩阵
r_doa=zeros(1,M*(M-1)/2);
%%  协方差矩阵上三角矩阵
k=1;
for i=1:M
    for j=i+1:M
        r_doa(k)=Rx(i,j);
        k=k+1;
    end
end

% r_doa=r_doa/norm(r_doa,2);
% r_doa1=1*[real(r_doa) imag(r_doa)];

% 
% r_doa=r_doa-min(real(r_doa))-1j*min(imag(r_doa));
r_doa=r_doa/norm(r_doa);
r_doa1=1*[real(r_doa) imag(r_doa)];
% temp=H'*vec(Rx);
% SS=temp/norm(temp);
% SS1=1*[real(SS) imag(SS)];
% ;
% r_doa=r_doa-mean(r_doa);
% r_doa=r_doa/norm(r_doa,2);
% r_doa1=[real(r_doa) imag(r_doa)];
%r_doa1=[angle(r_doa) abs(r_doa)].';


