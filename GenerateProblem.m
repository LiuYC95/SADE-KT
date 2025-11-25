function [A,B,C] = GenerateProblem(Xs,Xt,ys,yt,option)
% [ds,ns] = size(Ks);
% [dt,nt] = size(Kt);
% K = [Ks zeros(ds,nt);zeros(dt,ns) Kt];
% nt = length(Kt);
% ns = length(Ks);
% K = [Ks zeros(ns,nt);zeros(nt,ns) Kt];
theta = option.theta;
lambda=option.lambda;
alpha=0;%option.alpha;
beta=0;%option.beta;
gamma=0;%option.gamma;
Lwb_opt = option.Lwb_opt;
prelabel = option.prelabel;
classopt = option.classopt;
[dt,nt] = size(Xt);
[ds,ns] = size(Xs);
if theta~=0
    A = GenerateA(Xs,Xt);
    A = theta*A;
else
    A = 0;
end
X = [Xs zeros(ds,nt);zeros(dt,ns) Xt];
[E,L,Lw,Lb] = GenerateMatrix(Xs,Xt,ys,yt, Lwb_opt,prelabel,classopt);
B = X*(lambda(1)*E+ alpha*L + beta*Lw)* X';
C = gamma*X*Lb*X';
end

function A = GenerateA(Xs,Xt)
%generating X
[dt,nt] = size(Xt);
[ds,ns] = size(Xs);
Xs = centering(Xs);%sample center zero
Xt = centering(Xt);
A = [Xs*Xs'/ns zeros(ds,dt);zeros(dt,ds) -Xt*Xt'/nt];
end