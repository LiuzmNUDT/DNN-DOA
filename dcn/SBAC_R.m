



% SBAC (Sparse Bayesian Array Calibration) 方法函数
% 2012-09-03

function [gamma,mu] = SBAC_R(X,A,MaxItr,ErrorThr,enta,K)
[M,N]=size(X);
R=X*X'/N;
Qx=kron(R.',R)/N;
D=eig(R);
D=sort(D);
sigma2 =mean(D(1:end-K));
%% DcRVM
% (1) Initialize
R0=R-sigma2*eye(M);
x=vec(R0);
mu = A'*pinv( A* A')*(x);
gamma = Row2Norm_Fun(mu);
B = A;
% (2) Update
StopFlag = 0;
ItrIdx = 1;
% --- 空域独立 SBL 迭代至稳态解附近
while ~StopFlag && ItrIdx < MaxItr
%     err(ItrIdx)=norm(mu-enta,'fro')/norm(enta,'fro');
    
    gamma0 =  gamma;
    % 空间谱更新
    Q = Qx+B*diag(gamma)*B';
    Qinv = pinv(Q);
    Sigma = diag(gamma)-diag(gamma)*B'*Qinv*B*diag(gamma);   % 信号幅度估计协方差
    mu = diag(gamma)*B'*Qinv*x;                    % 信号幅度
    gamma = abs(Row2Norm_Fun(mu)+diag(Sigma));
    if norm( gamma- gamma0,'fro')/norm( gamma,'fro') < ErrorThr
        StopFlag = 1;
    else
        ItrIdx = ItrIdx+1;
    end
end





