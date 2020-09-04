
% SBAC (Sparse Bayesian Array Calibration) 方法函数
% 2012-09-03

function [gamma,mu] = SBAC(x,A,MaxItr,ErrorThr)
[M,NT] = size(x);

%% DcRVM
% (1) Initialize
mu = A'*pinv( A* A')*(x);
sigma2 = 0.1*norm(x,'fro')^2/(M*NT);
gamma = Row2Norm_Fun(mu);
B = A;
% (2) Update
StopFlag = 0;
ItrIdx = 1;
% --- 空域独立 SBL 迭代至稳态解附近
while ~StopFlag && ItrIdx < MaxItr
    gamma0 = gamma;
%     err(ItrIdx)=norm(mu-x0,'fro')/norm(x0,'fro');
    % 空间谱更新
    Q = sigma2*eye(M)+B*diag(gamma)*B';
    Qinv = pinv(Q);
    Sigma = diag(gamma)-diag(gamma)*B'*Qinv*B*diag(gamma);   % 信号幅度估计协方差
    mu = diag(gamma)*B'*Qinv*x;                     % 信号幅度
    sigma2 = real(((norm(x-B*mu,'fro'))^2+NT*trace(B*Sigma*B'))/(M*NT));
    gamma = abs(Row2Norm_Fun(mu)+diag(Sigma));
   
    if norm(gamma-gamma0)/norm(gamma)< ErrorThr
        StopFlag = 1;
    else
        ItrIdx = ItrIdx+1;
    end
end



