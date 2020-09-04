
% 对二维数据矩阵逐行取2范数（平方）

function gamma = Row2Norm_Fun(mu)

[L,N] = size(mu);
gamma=diag(mu*mu')/N;
end