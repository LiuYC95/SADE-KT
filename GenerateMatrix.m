function [E,L,Lw,Lb] = GenerateMatrix(Xs,Xt,ys,yt, Lwb_opt,prelabel,classopt)
[ds,ns] = size(Xs);
[dt,nt] = size(Xt);
class=unique(ys);
nclass=numel(class);
for i = 1: nclass
    ids = find(ys==class(i));
    idt = find(yt==class(i));
    yys(ids) = i;
    yyt(idt) = i;
end
ys = yys'; yt = yyt';
e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
E = e*e';
%E = E/norm(E,'fro');

%Laplacian
manifold.k = 5;   %??????????????????????????????????
manifold.Metric = 'Cosine';% 'Cosine'
manifold.NeighborMode = 'KNN';
manifold.WeightMode = 'Cosine';%'Cosine'
manifold.bNormalizeGraph = 1;
[Ws,Ds] = laplacian(Xs',manifold);
[Wt,Dt] = laplacian(Xt',manifold);
L=[Ds-Ws zeros(ns,nt);zeros(nt,ns) Dt-Wt];
y = [ys;yt];
if strcmp(classopt, 'all')
    Ww = sparse(ns+nt,ns+nt);
    Wb = sparse(ns+nt,ns+nt);
    for i = 1:length(class)
        id = find(y==i);
        idn = setdiff([(1:ns)';ns+select],id);
        Ww(id,id) = 1;
        Wb(id,idn) = 1;
    end
elseif strcmp(classopt,'st')
    Ww = eye(ns+nt,ns+nt);
    Wb = eye(ns+nt,ns+nt);
    for i = 1: nclass
        ids = find(ys ==i);
        idt = ns+find(yt ==i);
        idtn = setdiff(ns,idt);
        Ww(ids,idt) = 1;
        Ww(idt,ids) = 1;
        Wb(ids,idtn) = 1;
        Wb(idtn,ids) = 1;
    end
    % Ww = Ww/length(find(Ww(:)));
    % Wb = Wb/length(find(Wb(:)));
end
Lw = LAP(Ww);
Lb = LAP(Wb);
end

function L = normLAP(W)
n = length(W);
D = 1./sqrt(sum(W));
D(isinf(D)) = 0;
D=diag(sparse(D));
I=D;
I(I>0) = 1;
%I=sparse(1:n,1:n,1);
L = I-D*W*D;
end

function L = LAP(W)
n = length(W);
D = diag(sum(W));
L = D-W;
end