function [X_HKT,AAA] = HKT(DB,bu,bd,DataS,ii,stage,AAA)
%% KT1
c1 = size(DB,2)-1;
Data = DataS{ii};
c2 = size(Data,2)-1;
if ii == 1
    [~,bs] = sort(Data(1:330,c2+1));
end
if ii == 2
    [~,bs] = sort(Data(1:220,c2+1));
end
meanX = Data(bs(1),:);
bul = meanX;
bul(end:c1) = bu;
bdl = meanX;
bdl(end:c1) = bd;
p1 = unique(DB,'rows');
x = p1(:,1:c1);
y = p1(:,c1+1);
kernal='cubic';
L = x; LY =y;
global coefC ifu
coefC = rbfcreate(L',LY','RBFFunction', kernal);
nameR = @YCRBF;
[Nap,fap]=YPSO(@YCRBF,[bdl',bul']); %Nap is the best for the model

%% KT2
if ii == 1
    if stage == 2
        AA = DataS{1,1};
        Xs = AA(:,1:30)';
        ys = AA(:,31)';
        DD = DB;
        Xt = DD(:,1:50)';
        yt = DD(:,51)';
    end
    if stage == 3
        AA = DataS{1,1};
        Xs = AA(:,1:30)';
        ys = AA(:,31)';
        DD = DB;
        Xt = DD(:,1:100)';
        yt = DD(:,101)';
    end
end
if ii == 2
    AA = DataS{1,2};
    Xs = AA(:,1:50)';
    ys = AA(:,51)';
    DD = DB;
    Xt = DD(:,1:100)';
    yt = DD(:,101)';
end
[~,ns] = size(ys);
[~,nt] = size(yt);
num_ranklabels = min(ns,nt);
% num_ranklabels = 100;
X_s = Xs;
X_t = Xt;
f_s = ys;
f_t = yt;
y_s = fit_relax(f_s,num_ranklabels);
y_t = fit_relax(f_t,num_ranklabels);
[X_sn,means,stds] = zscore(X_s');
[X_tn,meant,stdt] = zscore(X_t');
X_sa = X_sn';
X_ta = X_tn';
[ds,ns] = size(X_sa);
[dt,nt] = size(X_ta);
% mapping training
alpha = 0.1;
d_low = 4;
% d_low = 3;
T_max = 100;
tol = 1e-9;
X_total = [X_sa zeros(ds,nt);zeros(dt,ns) X_ta];
A = [X_sa*X_sa'/ns zeros(ds,dt);zeros(dt,ds) -X_ta*X_ta'/nt];
Ls = LaplacianMatrix(y_s,y_t);
B = X_total*(alpha*Ls)*X_total';
[Vb,Db] = eig(B);
[~,indb] = sort(diag(Db));
P = real(Vb(:,indb(1:d_low)));
T = 1;
floss_old = min(ns,nt);
while T<T_max
    floss = norm(P'*A*P,'fro')+trace(P'*B*P);
    if norm(floss-floss_old,2)<tol*floss
        break;
    end
    M = A*P*P'*A+1/2*B;
    [Vm,Dm] = eig(M);
    [~,indm] = sort(diag(Dm));
    floss_old = floss;
    P = real(Vm(:,indm(1:d_low)));
    T = T+1;
end

xlb = repmat(bd,1,dt);
xub = repmat(bu,1,dt);
% solution transfer
Ps1 = P(1:ds,:);
Pt1 = P(ds+1:end,:);
if ii == 1
    [BB,CC] = min(AA(1:11*ds,ds+1));
end
if ii == 2
    [BB,CC] = min(AA(1:11*20,ds+1));
end
BestS = AA(CC,1:ds);
xs = (BestS-means)./stds;
xs_trann = transpose((Pt1*Pt1')\Pt1*(Ps1'*xs'));
xNew = xs_trann.*stdt+meant;
xNew(xNew>xub)=xub(xNew>xub);
xNew(xNew<xlb)=xlb(xNew<xlb);
Pre_val = feval(nameR,xNew);

%% Choose the predicted one
if Pre_val < fap
    X_HKT = xNew; AAA = [AAA,1];
else
    X_HKT = Nap; AAA = [AAA,0];
end
end