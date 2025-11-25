% Yuanchao Liu, Jianchang Liu, Jinliang Ding, Shangshang Yang, Yaochu Jin. A surrogate-assisted differential evolution with knowledge 
%transfer for expensive incremental optimization problems[J].IEEE Transactions on Evolutionary Computation, 2024, 28(4): 1039-1053.

warning off;
clc;
clear all;
tic;
format long;
format compact;
clearvars -except data l t t1;
for problemIndex = 1
    clearvars -except problemIndex;
    for l = 1:21
        clearvars -except data l t t1 problemIndex;
        mu = 70;
        ifu = problemIndex;
        DataS = [];
        AA = [];
        if ifu==1
            %Cheng et al. [56] in 2019
            name=@imop;
            global func_num
            func_num=1;
            global bu bd
            bu=100; bd=-100; vmax=bu; vmin=bd;
            design = imop(func_num);
        end
        if ifu==2
            
            name=@imop;
            global func_num
            func_num=2;
            global bu bd
            bu=5; bd=-5; vmax=bu; vmin=bd;
            design = imop(func_num);
        end
        if ifu==3
            name=@imop;
            global func_num
            func_num=3;
            global bu bd
            bu=100; bd=-100; vmax=bu; vmin=bd;
            design = imop(func_num);
        end
        if ifu==4
            
            name=@imop;
            global func_num
            func_num=4;
            global bu bd
            bu=5; bd=-5; vmax=bu; vmin=bd;
            design = imop(func_num);
        end
        if ifu==5
            
            name=@imop;
            global func_num
            func_num=5;
            global bu bd
            bu=100; bd=-100; vmax=bu; vmin=bd;
            design = imop(func_num);
        end
        if ifu==6
            
            name=@imop;
            global func_num
            func_num=6;
            global bu bd
            bu=5; bd=-5; vmax=bu; vmin=bd;
            design = imop(func_num);
        end
        if ifu==7
            
            name=@imop;
            global func_num
            func_num=7;
            global bu bd
            bu=5; bd=-5; vmax=bu; vmin=bd;
            design = imop(func_num);
        end
        Rexcl = [];
        k = 1;
        POP = [];
        cs = [];
        for stage = 1:3
            DB = [];
            if stage == 1
                cs(stage) = 30;
                termination = 11*cs(stage);
            end
            if stage == 2
                cs(stage) = 50;
                termination = 11*(cs(stage)-cs(stage-1));
            end
            if stage == 3
                cs(stage) = 100;
                termination = 11*(cs(stage)-cs(stage-1));
            end
            func = @(x)(imop(x', stage));
            % Define the dimension of the problem
            n = cs(stage);
            popsize = 10;
            Nsample = 100;
            % Set the population size
            % Define the boundary
            lu = [bd* ones(1, n); bu* ones(1, n)];
            rand('seed', sum(100 * clock));
            
            % Initialize the main population
            x = repmat((lu(2, :) - lu(1, :)), Nsample, 1) .* lhsdesign(Nsample, n) + repmat(lu(1, :), Nsample, 1);

            y = func(x);
            y = y';

            DB=[x,y];
            
            [~,b]=sort(y);
            popold = x(b(1:popsize),:);
            valParents = y(b(1:popsize));
            
            c = 1/10;
            p = 0.05;
            
            CRm = 0.5;
            Fm = 0.5;
            
            % Initialize the selection ration in the Eigen coordiante system as sita = (0.5, 0.5, ...0.5).
            sita = 0.5.*ones(1,popsize);
            
            % Record the values and indices of the best solutions
            [~, indBest] = sort(valParents, 'ascend');
            
            % Compute the weights and mueff
            cov_length = popsize./2;
            weights = log(cov_length + 0.5) - log(1 : cov_length)';
            mueff = sum(weights)^2/sum(weights.^2);
            weights = weights/sum(weights);
            
            % Initialize the weighted mean value of the distribution: p_center
            p_sort = popold(indBest(1:popsize./2,:),:);
            p_center =(p_sort' * weights)';
            
            %Initialzie the archive population and fitness value in SVM-EA
            pop_pool = popold;
            fit_pool = valParents;
            center_pool = p_center;
            
            % Initialize covariance matrix C and B as unity matrix.
            B = eye(n);
            D = eye(n);
            BD = B * D;
            C = BD * transpose(BD);
            FES = Nsample;
            % The arvhive population in JADE itself
            popachieve = [];
            fitachieve = [];
            G = 0;
            while FES <  termination
                if stage == 1
                    POP.s1{G+1} = [popold, valParents];
                end
                if stage == 2
                    POP.s2{G+1} = [popold, valParents];
                end
                if stage == 3
                    POP.s3{G+1} = [popold, valParents];
                end
                
                
                
                [DB,FES,k] = LocalSearch(@imop,DB,FES,bu,bd,k,stage);
                
                [as,bs] = sort(valParents);
                DBL = DB(end,:);
                if as(end) > min(DBL(:,1+n))
                    [~,bD] = min(DBL(:,1+n));
                    popold(bs(end),:) = DBL(bD(1),1:n);
                    valParents(bs(end)) = min(DBL(:,1+n));
                end
                if stage ~= 1
                    for ii = 1:stage-1
                        [x_HKT,AA] = HKT(DB,bu,bd,DataS,ii,stage,AA);
                        y_HTK = func(x_HKT);
                        y_HTK = y_HTK';
                        S_HTK = [x_HKT,y_HTK];
                        DB = [DB;S_HTK];
                        FES = FES + 1;
                        [as,bs] = sort(valParents);
                        DBL = DB(end,:);
                        if as(end) > min(DBL(:,1+n))
                            [~,bD] = min(DBL(:,1+n));
                            popold(bs(end),:) = DBL(bD(1),1:n);
                            valParents(bs(end)) = min(DBL(:,1+n));
                        end
                    end
                end
                
                DB1 = unique(DB,'rows');
                ModelX = DB1(:,1:n);
                YModelX = DB1(:,n+1);
                
                kernal='cubic';
                global coefC
                coefC=[];
                coefC=rbfcreate(ModelX',YModelX','RBFFunction', kernal);
                nameR = @YCRBF;
                % The old population becomes the current population
                pop = popold;
                valTemp  = valParents;
                ui1 = [];
                u2 = [];
                ui = [];
                % Generate CR according to a normal distribution with mean CRm, and std 0.1
                % Generate F according to a cauchy distribution with location parameter Fm, and scale parameter 0.1
                [F_original, CR_original] = randFCR(popsize, CRm, 0.1, Fm, 0.1);
                [F_Eigen, CR_Eigen] = randFCR(popsize, CRm, 0.1, Fm, 0.1);
                
                % The general population
                popAll = [pop; popachieve];
                
                %==============================================================
                % ============In the original coordinate system================
                %==============================================================
                
                r0 = [1: popsize];
                [~, indBest] = sort(valParents, 'ascend');
                % Find the p-best solutions
                pNP = max(round(p * popsize), 2); %choose at least two best solutions
                randindex = ceil(rand(1, popsize) * pNP); %select from [1, 2, 3, ..., pNP]
                randindex = max(1, randindex); %to avoid the problem that rand = 0 and thus ceil(rand) = 0
                pbest = pop(indBest(randindex), :); % randomly choose one of the top 100p% solutions
                
                %  = == == == == == == == ==Mutation == == == == == == == == ==
                for i = 1:popsize
                    U = [];
                    R=randi([1,popsize],mu,3);
                    V1=pop(R(:,1),:)+repmat(F_original(i),mu,n).*(pop(R(:,2),:)-pop(R(:,3),:));
                    t=rand(mu,n)<=repmat(CR_original(i),1,n);
                    jrand=randi([1,n],1,mu)';
                    jrand_index=n*[0:mu-1]'+jrand;
                    t(jrand_index)=1;
                    U1=t.*V1+(1-t).*repmat(pop(i,:),mu,1);
                    U=[U;U1];
                    
                    R=randi([1,popsize],mu,3);
                    V2=pop(i,:) + repmat(F_original(i),mu,n).*(pop(i, :) - pop(R(:,1),:)) + repmat(F_original(i),mu,n).*(pop(R(:,2),:)-pop(R(:,3),:));
                    U2 = V2;
                    U=[U;U2];
                    
                    R=randi([1,popsize],mu,2);
                    V3=pbest(i,:) + repmat(F_original(i),mu,n).*(pop(R(:,1),:)-pop(R(:,2),:));
                    t=rand(mu,n)<=repmat(CR_original(i),1,n);
                    jrand=randi([1,n],1,mu)';
                    jrand_index=n*[0:mu-1]'+jrand;
                    t(jrand_index)=1;
                    U3=t.*V3+(1-t).*repmat(pop(i,:),mu,1);
                    U=[U;U3];
                    
                    R=randi([1,popsize],mu,2);
                    V4=pbest(i,:) + repmat(F_original(i),mu,n).*(pbest(i,:) - pop(i,:)) + repmat(F_original(i),mu,n).*(pop(R(:,1),:)-pop(R(:,2),:));
                    t=rand(mu,n)<=repmat(CR_original(i),1,n);
                    jrand=randi([1,n],1,mu)';
                    jrand_index=n*[0:mu-1]'+jrand;
                    t(jrand_index)=1;
                    U4=t.*V4+(1-t).*repmat(pop(i,:),mu,1);
                    U=[U;U4];
                    U = boundConstraint(U,lu);
                    
                    fitness = feval(nameR,U);
                    [~,b] = min(fitness);
                    ui1(i,:) = U(b(1),:);
                end
                
                % == == == == == == == Boundary repair == == == == == == == ==
                ui1 = boundConstraint(ui1,lu);
                
                %==============================================================
                % ============In the Eigen coordinate system===================
                %==============================================================
                
                r0 = [1 : popsize];
                [~, indBest] = sort(valParents, 'ascend');
                
                % Find the p-best solutions
                pNP = max(round(p * popsize), 2); % choose at least two best solutions
                randindex = ceil(rand(1, popsize) * pNP); % select from [1, 2, 3, ..., pNP]
                randindex = max(1, randindex); % to avoid the problem that rand = 0 and thus ceil(rand) = 0
                pbest = pop(indBest(randindex), :); % randomly choose one of the top 100p% solutions
                
                % == == == == == == == == == Mutation == == == == == == == == ==
                
                for i = 1:popsize
                    
                    
                    U = [];
                    R=randi([1,popsize],mu,3);
                    V1=pop(R(:,1),:)+repmat(F_original(i),mu,n).*(pop(R(:,2),:)-pop(R(:,3),:));
                    t=rand(mu,n)<=repmat(CR_original(i),1,n);
                    jrand=randi([1,n],1,mu)';
                    jrand_index=n*[0:mu-1]'+jrand;
                    t(jrand_index)=1;
                    J_= mod(floor(rand(mu, 1)*n), n) + 1;
                    J = (J_-1)*mu + (1:mu)';
                    vi2 = V1 * B;
                    ui2 = repmat(pop(i,:),mu,1) * B;
                    ui2(J) = vi2(J);
                    ui2=t.*vi2+(1-t).*ui2;
                    U1 = ui2 * inv(B);
                    U=[U;U1];
                    
                    R=randi([1,popsize],mu,3);
                    V2=pop(i,:) + repmat(F_original(i),mu,n).*(pop(i, :) - pop(R(:,1),:)) + repmat(F_original(i),mu,n).*(pop(R(:,2),:)-pop(R(:,3),:));
                    J_= mod(floor(rand(mu, 1)*n), n) + 1;
                    J = (J_-1)*mu + (1:mu)';
                    vi2 = V2 * B;
                    ui2 = repmat(pop(i,:),mu,1) * B;
                    ui2(J) = vi2(J);
                    U2 = ui2 * inv(B);
                    U=[U;U2];
                    
                    R=randi([1,popsize],mu,2);
                    V3=pbest(i,:) + repmat(F_original(i),mu,n).*(pop(R(:,1),:)-pop(R(:,2),:));
                    t=rand(mu,n)<=repmat(CR_original(i),1,n);
                    jrand=randi([1,n],1,mu)';
                    jrand_index=n*[0:mu-1]'+jrand;
                    t(jrand_index)=1;
                    J_= mod(floor(rand(mu, 1)*n), n) + 1;
                    J = (J_-1)*mu + (1:mu)';
                    vi2 = V3 * B;
                    ui2 = repmat(pop(i,:),mu,1) * B;
                    ui2(J) = vi2(J);
                    ui2=t.*vi2+(1-t).*ui2;
                    U3 = ui2 * inv(B);
                    U=[U;U3];
                    
                    R=randi([1,popsize],mu,2);
                    V4=pbest(i,:) + repmat(F_original(i),mu,n).*(pbest(i,:) - pop(i,:)) + repmat(F_original(i),mu,n).*(pop(R(:,1),:)-pop(R(:,2),:));
                    t=rand(mu,n)<=repmat(CR_original(i),1,n);
                    jrand=randi([1,n],1,mu)';
                    jrand_index=n*[0:mu-1]'+jrand;
                    t(jrand_index)=1;
                    J_= mod(floor(rand(mu, 1)*n), n) + 1;
                    J = (J_-1)*mu + (1:mu)';
                    vi2 = V4 * B;
                    ui2 = repmat(pop(i,:),mu,1) * B;
                    ui2(J) = vi2(J);
                    ui2=t.*vi2+(1-t).*ui2;
                    U4 = ui2 * inv(B);
                    U=[U;U4];
                    U = boundConstraint(U,lu);
                    fitness=feval(nameR,U);
                    [~,b]=min(fitness);
                    u2(i,:) = U(b(1),:);
                end
                
                
                % == == == == == == == Boundary repair == == == == == == == ==
                ui2 = boundConstraint(u2, lu);
                
                % Select the coordiante system based on the selection vecotr sita to implement the offspring
                system_selected = zeros(1,popsize);
                for i = 1 : popsize
                    if rand<sita(i)
                        ui(i,:) = ui1(i,:);
                        F_new(i,:) = F_original(i,:);
                        CR_new(i,:) = CR_original(i,:);
                        system_selected(i) = 1;
                    else
                        ui(i,:) = ui2(i,:);
                        F_new(i , :) = F_Eigen(i,:);
                        CR_new( i,:) = CR_Eigen(i,:);
                    end
                end
                
                % Evaluation the trial population
                

                valOffspring = func(ui);
                valOffspring = valOffspring';

                
                DB1 = [ui,valOffspring];
                DB=[DB;DB1];
                FES = FES + popsize;
                
                % Set the archive popsize as T*popsize
                T  = 3;
                
                % Update the archieve population based on the rule of "first-in and first-out"
                pop_pool = [pop_pool;ui];
                fit_pool = [fit_pool;valOffspring];
                
                if size(pop_pool,1) > T*popsize
                    pop_pool(1:popsize,:) = [];
                    fit_pool(1:popsize,:) = [];
                end
                
                G = G + 1;
                
                % Sort the individul in the archive population in ACoS
                [~, index2] = sort(fit_pool);
                mutant_sort = pop_pool(index2,:);
                
                % One half best individual is selected
                cov_length= size(mutant_sort,1)./2;
                p_select =  mutant_sort(1 : cov_length,:);
                
                % Compute the weights and mueff
                weights = log(cov_length + 0.5) - log(1 : cov_length)';
                mueff = sum(weights)^2/sum(weights.^2);
                weights = weights/sum(weights);
                
                % Compute the learning ratio cmu
                cmu =min(mueff./n^2,1);
                
                % Employ the Rank-mu-update method to estimate C_u
                p_to_mean =  p_select - p_center(ones(1,cov_length),:);
                C_u =p_to_mean'*diag(weights)*p_to_mean;
                
                % Utilize the cumulative information to estimate C
                C = (1 - cmu) * C + cmu .* C_u;
                C = triu(C) + triu(C, 1)';
                
                % Eigen decomposition of C
                if mod(G,3) == 0
                    [B, D] = eig(C);
                    % limit condition of C to 1e20 + 1
                    if max(diag(D)) > 1e20*min(diag(D))
                        tmp = max(diag(D))/1e20 - min(diag(D));
                        C = C + tmp*eye(n);
                        [B, D] = eig(C);
                    end
                end
                
                %  Compute the weighted center of population distribution
                p_center = (p_select' * weights)';
                
                % == == ==   == == ==  == ==Selection  == == ==  == == == == ==
                F_success = [];
                CR_success = [];
                
                pcount_achi = [];
                fitcount_achi = [];
                
                sita_success= [];
                value_differ = [];
                count_achi = 0;
                number_success = [];
                
                success_record = zeros(1,popsize);
                
                for i = 1:popsize
                    if valTemp(i) >= valOffspring(i)
                        pop(i,:) = ui(i,:);
                        valParents(i,:) = valOffspring(i,:);
                        
                        count_achi = count_achi + 1;
                        pcount_achi(count_achi,:) = popold(i,:);
                        fitcount_achi(count_achi,:) = valTemp(i,:);
                        
                        F_success(count_achi) = F_new(i);
                        CR_success(count_achi) = CR_new(i);
                        
                        number_success(count_achi) = system_selected(i);
                        
                        success_record(i) = 1;
                    end
                end
                popold = pop;
                valTemp = valParents;
                % Update Fm and CRm
                if count_achi~= 0
                    goodF = F_success;
                    goodCR = CR_success;
                    
                    CRm = (1 - c) * CRm + c * mean(goodCR);
                    Fm = (1 - c) * Fm + c * sum(goodF .^ 2) / sum(goodF);
                end
                % Upadate  the probability vector sita
                para = 0.05;
                alfa = 2;
                for i = 1:popsize
                    if system_selected(i)==1 && success_record(i) == 1
                        sita(i) = sita(i) +para*(1-sita(i))*exp(-alfa*sita(i));
                    elseif system_selected(i)==0 && success_record(i) == 1
                        sita(i) = sita(i) - para*(sita(i))*exp(alfa*sita(i)-alfa*1);
                    elseif system_selected(i)==1 &&success_record(i) ==0
                        sita(i) = sita(i) -0.1* para*(1-sita(i))*exp(-alfa*sita(i));
                    else
                        sita(i) = sita(i) + 0.1*para*(sita(i))*exp(alfa*sita(i)-alfa*1);
                    end
                end
                sita = max(min(sita,1),0);
                %  Archive mechanism in JADE itself
                [popachieve, fitachieve] = update_achive(popachieve, fitachieve, pcount_achi, fitcount_achi, popsize);
                FES
                min(DB(:,1+n))
            end
            DataS{stage} = DB;
        end
        data{l} = DataS;
    end
end
% save F9 data
for i = 1:21
    AA = data{i};
    for j = 1:3
        if j == 1
            BB(i,j) = min(AA{j}(1:11*30,30+1));
        end
        if j == 2
            BB(i,j) = min(AA{j}(1:11*20,50+1));
        end
        if j == 3
            BB(i,j) = min(AA{j}(1:11*50,100+1));
        end
    end
end
meanB = mean(BB)
stdB = std(BB)
toc;