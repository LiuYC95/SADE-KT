function [ Y ] = YCRBF(T )
%MYL3 Summary of this function goes here
% global coefC
% Y=rbfinterp(T', coefC);
% Y=Y';


global coefC
Y=rbfinterp(T', coefC);
end

