function [p_achive, fit_achive] = update_achive(p_achive, fit_achive, pcount_achi, fitcount_achi, popsize)

if size(pcount_achi, 1) == 0, return; end

popAll = [p_achive; pcount_achi];
funvalues = [fit_achive; fitcount_achi];

[~, IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
    popAll = popAll(IX, :);
    funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= popsize   % add all new individuals
    p_achive = popAll;
    fit_achive = funvalues;
    
else                % randomly remove some solutions
    rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
    rndpos = rndpos(1 : popsize);
    
    p_achive = popAll(rndpos, :);
    fit_achive = funvalues(rndpos, :);
    
end
