function vi = boundConstraint (vi, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang, jingqiao@gmail.com

[NP, D] = size(vi);  % the population size and the problem's dimension

%% check the lower bound
xl = repmat(lu(1, :), NP, 1);
pos = vi < xl;
vi(pos) = xl(pos) + rand;

%% check the upper bound
xu = repmat(lu(2, :), NP, 1);
pos = vi > xu;
vi(pos) = xu(pos) - rand;