function [ T ] = Rastring( X )
%RASTRING Summary of this function goes here
%   Detailed explanation goes here
Dim = size(X,2);
% load rastrigin_func_data;%%%%%%%%%%%
% if Dim==10
%     load rastrigin_M_D10;
% end
% if Dim==30
%     load rastrigin_M_D30;
% end
% if Dim==50
%     load rastrigin_M_D50;
% end
% X=bsxfun(@minus,X,o(1:Dim))*M;
T=0;
for i=1:Dim
    T=T+X(:,i).^2-10*cos(2*pi*X(:,i))+10;
end
% T=T-330;

end

