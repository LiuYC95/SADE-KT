function [DB,FES,k] = LocalSearch(imop,DB,FES,bu,bd,k,stage)
c = size(DB,2)-1;
nl = 5*c;
D1 = sortrows(DB,c+1);
if length(D1) < 5*c
    D1 = D1;
else
    D1 = D1(1:5*c,:);
end
func = @(x)(imop(x', stage));
% if k == 1
    [~,xmin]=min(D1(:,c+1));
    Xmin=D1(xmin,:);
    Xmin = Xmin(:,1:c);
    [~,xmax]=max(D1(:,c+1));
    Xmax=D1(xmax,:);
    Xmax = Xmax(:,1:c);
    Rexcl=norm(Xmin-Xmax)/2;
% end
% k =  popsize/2;

% for i = 1:k
    Pl = sortrows(DB,c+1);
    meanX = Pl(1,:);
    bul = meanX(:,1:c) + Rexcl;
    bul(bul > bu) = bu;
    bdl = meanX(:,1:c) - Rexcl;
    bdl(bdl < bd) = bd;
    %% 选取合适的点建立局部模型
    Pl = unique(Pl,'rows');
    D=Pl;
    a111 = [];
    for i = 1:size(D,1)
        distance = sum(abs(Pl(:,1:c)-repmat(Pl(i,1:c),size(Pl,1),1)).^2,2).^(1/2);
        a1111=find(distance < 0.001);
        a1111(find(a1111==i)) = [];
        a111=[a111;a1111];
    end
    Pl(a111,:) = [];
    distance = sum(abs(Pl(:,1:c)-repmat(meanX(:,1:c),size(Pl,1),1)).^2,2).^(1/2);
    [a1, b]= sortrows(distance,1);
    a11 = find(a1<Rexcl);
    if size(Pl,1) < nl %+ 0.1*nl
        Datal = Pl;
    else if length(a11) < nl
            Datal = Pl(b(1:nl),:);
        else
            Datal = Pl(a11,:);
        end
    end

    p1 = Datal;
    p1 = unique(p1,'rows');
    x = p1(:,1:c);
    y = p1(:,c+1);
    
    kernal='cubic';
    L = x; LY =y;
    global coefC
    coefC = rbfcreate(L',LY','RBFFunction', kernal);
    nameR = @YCRBF;
    [Nap,fap]=YPSO(@YCRBF,[bdl',bul']); %Nap is the best for the model  
    ifu = 1;
    
    RealFitness = func(Nap);
    RealFitness = RealFitness';
    
%     RealFitness = feval(name,Nap);
    FES = FES + 1;
    gbest = [Nap,RealFitness];
    DB = [DB;gbest];
    gbestf = fap;
    gbestxf = feval(nameR,meanX(1,1:c));
    
    ro=(meanX(:,c+1)-gbest(:,c+1))/(gbestxf - gbestf-10^(-20));
    if norm(meanX(:,1:c)-gbest(:,1:c)) < Rexcl
        es=1;
    end
    if norm(meanX(:,1:c)-gbest(:,1:c)) >= Rexcl
        es=2;
    end

    k = k + 1;
    
    
% end
end