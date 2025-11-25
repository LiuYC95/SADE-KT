function [ T ] = Rastrigin( X )
%Rastrigin Summary of this function goes here
global Dim
T=10*Dim+sum(X.^2-10*cos(2*pi*X),2);
end

