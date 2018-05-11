function [expected_theta,expected_beta,expected_tau,expected_phi, gamma,prior,cuv] =...
    IIRPF(events,eventsMatrix,trinUserEvents,...
        T,t0,inedges,outedges,prior,params,kernel,likelihoodIsRequired,startIteration)
U = params.U;
P = params.P;
K = params.K;
datasetName = params.datasetName;
methodName = params.methodName;
maxNumberOfIterations = params.maxNumberOfIterations;
saveInterval = params.saveInterval;
plottingInIteration = params.plottingInIteration;
if nargin<15
    startIteration = 0;
end
if nargin<14
    likelihoodIsRequired = 0;
end
if nargin<13
    w=1;
%     g = @(x,w) w*exp(-w*x);
%     g_factorized = @(x,w,p) p*exp(-w*x);
    g_log = @(x,w) log(w)-w*x;
    G = @(x,w) 1-exp(-w*x);
else
    w = kernel.w;
%     g = kernel.g;
%     g_factorized = kernel.g_factorized;
    G = kernel.G;
    g_log = kernel.g_log;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%       compute B Values          %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%    B(u,v)= \sum_{e\in H_u(t_0(v)^+} G(T-t_e) %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SUMG_User = zeros(U,U);
for u=1:U
    for p=1:P
        for i=1:length(eventsMatrix{u,p})
            te = eventsMatrix{u,p}(i);
            for v=outedges{u}
                SUMG_User(u,v) = SUMG_User(u,v)+ G(T-te,w);
                if (t0(v) > te)
                    SUMG_User(u,v) = SUMG_User(u,v) - G(t0(v)-te,w);
                end
            end            
        end
    end
end

SUMG_User_Product = zeros(U,P);
for n=1:length(events)
    un = events{n}.user;
    pn = events{n}.product;
    tn = events{n}.time;
    SUMG_User_Product(un,pn) =  SUMG_User_Product(un,pn)+G(T-tn,w);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if startIteration ==0
    rng(0);
    gamma = initializeGamma(inedges, outedges, prior,params);
    iteration = 0;
else
    iteration = startIteration;
    load(sprintf('../../../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,startIteration));
end
plotDataCount = maxNumberOfIterations;  
x = 1:maxNumberOfIterations;
elboList = zeros(maxNumberOfIterations,1);
if startIteration>0
    elboList(startIteration) = -inf;
end
logLikelihoodList = zeros(maxNumberOfIterations,1);
% oldLogLikelihood = -inf;
if likelihoodIsRequired
    [expected_theta,  expected_beta, expected_tau, expected_phi] = ...
    estimateParams(gamma,inedges,U);

    likelihood = logLikelihoodEstimator(events, outedges,...
                        expected_tau,expected_theta,expected_beta, expected_phi, ...
                    SUMG_User, params, ...
                        kernel, T,t0);
    fprintf('loglikelihood is %.2f\n',likelihood);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (iteration<maxNumberOfIterations)% && newLogLikelihood-oldLogLikelihood > 1e-5)
    iteration = iteration+1;
    fprintf('Iteration %d\n',iteration)
    tic
    [cuv, cuk, cpk,cpqk,cu_bar,sum_entropy_multinomial,sum_sij_gw] = ...
    quadraticTimeComputeS(events, trinUserEvents,...
        gamma, w, g_log, params);
    fprintf('1-S parameters updated. Elapsed Time is %f\n',toc);
    tic
    [gamma] = updateTau(U,inedges,prior,cuv,SUMG_User,gamma);   
    fprintf('2-Tau Parameters Updated. Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updateTheta(K,cuk,prior,gamma, T,t0);
    fprintf('3-Theta Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updateBeta(P,K,cpk,prior,gamma, T,t0,SUMG_User_Product,...
    cpqk);
    fprintf('4-Beta Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updateEta(U,prior,gamma);
    fprintf('5-Eta Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updateMu(U,outedges,prior,gamma);
    fprintf('6-Mu Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updateKsi(P,prior,gamma);
    fprintf('7-Ksi Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updatePsi(prior,gamma);
    fprintf('8-Psi Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updatePhi(P,prior, gamma, cu_bar, SUMG_User_Product);
    fprintf('9-Phi Parameters Updated.Elapsed Time is %f\n',toc);
%%%%%%%%%%%% Saving Elbo and LogLikelihood %%%%%%%%%%%%%%%%%
    tic;
    elboList(iteration) = ELBO(U,P,K,T,inedges,SUMG_User,SUMG_User_Product,...
            sum_sij_gw,sum_entropy_multinomial,...
            prior,...
            cuk,cpk,cuv,cpqk,cu_bar,t0,gamma);
    fprintf('Elbo Computed. Elapsed Time is %f\n',toc);
    if likelihoodIsRequired
        tic;
        [expected_theta, expected_beta, ...
        expected_tau, expected_phi] = ...
        estimateParams(gamma,inedges,U);
        likelihood = logLikelihoodEstimator(events, outedges,...
                        expected_tau,expected_theta,expected_beta, expected_phi, SUMG_User, params, ...
                        kernel, T,t0);
        fprintf('LogLikelihood Computed.Elapsed Time is %f\n',toc);
        logLikelihoodList(iteration) = likelihood; 
    end
    
    if iteration>1
        if (elboList(iteration)-elboList(iteration-1) < -1e-5)
            fprintf('BUG ELBO: diffElbo = %f',elboList(iteration)-elboList(iteration-1));
            break;
        end
        fprintf('%d\t DeltaElbo=%f\t deltaLL=%f. \n',iteration,elboList(iteration)-elboList(iteration-1),...
            logLikelihoodList(iteration)-logLikelihoodList(iteration-1));
    end
    if mod(iteration,saveInterval) ==0
        [expected_theta, expected_beta, ...
        expected_tau, expected_phi] = estimateParams(gamma,inedges,U);
       save(sprintf('../../../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,iteration),...
        'expected_theta','expected_beta','expected_tau','expected_phi','kernel','params','prior','gamma','cuv');
    end
    
%%  Plotting Results
    if (plottingInIteration == 1)
        subplot(3,1,1);
        plot(x(1:plotDataCount),elboList(1:plotDataCount));
        subplot(3,1,2);
        plot(x(1:plotDataCount),logLikelihoodList(1:plotDataCount));
        stem(x(10:plotDataCount),diffElboBeta(10:plotDataCount)); 
        subplot(3,1,3);
        stem(x(10:plotDataCount),diffElboKsi(10:plotDataCount)); 
        drawnow
    end
end
[expected_theta,  expected_beta, expected_tau, expected_phi] = estimateParams(gamma,inedges,U);
save(sprintf('../../../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,maxNumberOfIterations),...
    'expected_theta','expected_beta','expected_tau','expected_phi','kernel','params','prior','gamma','cuv');
end

function [expected_theta,  expected_beta, ...
    expected_tau, expected_phi] = estimateParams(gamma,inedges,U)
    expected_beta = gamma.beta_shp ./ gamma.beta_rte;
    expected_theta = gamma.theta_shp ./ gamma.theta_rte;
    expected_tau = zeros(U,U);
    for v=1:U
        expected_tau(inedges{v},v) = gamma.tau_shp(inedges{v},v)./gamma.tau_rte(inedges{v},v);
    end
    expected_phi = gamma.phi_shp ./ gamma.phi_rte;
end
%% 2.2 theta
function [gamma] = updateTheta(K,cuk,prior,gamma, T,t0)
    sum_expected_beta_p = squeeze(sum(gamma.beta_shp./gamma.beta_rte,1)); % 1*K matrix
    gamma.theta_shp = prior.shape.theta+cuk;
    gamma.theta_rte = (T-t0) * sum_expected_beta_p +repmat(gamma.eta_shp./gamma.eta_rte,[1,K]);
end
%% 2.3 beta
function [gamma,result] = updateBeta(P,K,cpk,prior,gamma, T,t0,SUMG_User_Product,cpqk)

    result = 0;
    sum_expected_theta_u = (T-t0)'*(gamma.theta_shp./gamma.theta_rte); %(1*U) * (U*K) = 1*K
    expected_phi = gamma.phi_shp./gamma.phi_rte;
    expected_ksi = gamma.ksi_shp./gamma.ksi_rte;
    expected_beta = gamma.beta_shp ./ gamma.beta_rte;
    sum_expected_beta = sum(expected_beta); %1*K
    phi_weighted_SUMG_product = sum(SUMG_User_Product.*repmat(expected_phi,1,P));% 1*P
    for k=1:K
        for p=1:P
%             fprintf('Diff is %f',sum(cpqk(p,:,k))+sum(cpqk(:,p,k))-sum_cpqk(p,k));
            gamma.beta_shp(p,k) = prior.shape.beta+cpk(p,k)+sum(cpqk(p,:,k))+sum(cpqk(:,p,k));%sum_cpqk(p,k);
            gamma.beta_rte(p,k) = sum_expected_theta_u(k)+expected_ksi(p)+...
                (sum_expected_beta(k)-expected_beta(p,k))*expected_phi'*SUMG_User_Product(:,p)+...
                phi_weighted_SUMG_product*expected_beta(:,k)-phi_weighted_SUMG_product(p)*expected_beta(p,k);
            % update sum expected
            
            new_expected_beta = gamma.beta_shp(p,k)/gamma.beta_rte(p,k);
            sum_expected_beta(k) = sum_expected_beta(k)-expected_beta(p,k)+new_expected_beta;
            expected_beta(p,k) = new_expected_beta;
        end
    end
end
%% === 2.1 tau
function [gamma] = updateTau(U,inedges,prior,cuv,B,gamma)
    for v=1:U        
        for u=inedges{v} %u influences on v
            gamma.tau_shp(u,v) = prior.shape.tau+cuv(u,v); %c(u,v)
            gamma.tau_rte(u,v) = B(u,v) + gamma.mu_shp(u)/gamma.mu_rte(u);
        end
    end
end
%% 2.5 eta
function [gamma] = updateEta(U,prior,gamma)
    for u=1:U
        user_expected_theta =  sum(gamma.theta_shp(u,:) ./ gamma.theta_rte(u,:));
        gamma.eta_rte(u) = prior.rate.eta + user_expected_theta;
    end
end
%% 2.6 mu
function [gamma] = updateMu(U,outedges,prior,gamma)
    for u=1:U
        user_expected_tau = sum(gamma.tau_shp(u,outedges{u})./gamma.tau_rte(u,outedges{u}));
        gamma.mu_rte(u) = prior.rate.mu + user_expected_tau;
    end
end
%% 2.7 ksi
function [gamma] = updateKsi(P,prior,gamma)
    for p=1:P
        product_expected_beta = sum( gamma.beta_shp(p,:) ./ gamma.beta_rte(p,:));
        gamma.ksi_rte(p) = prior.rate.ksi + product_expected_beta;
    end
end

%% psi
function [gamma] = updatePhi(P, prior,gamma,cu_bar ,SUMG_User_Product)
    expected_beta = gamma.beta_shp./gamma.beta_rte;
    sum_expected_beta = sum(expected_beta);
    alpha = zeros(P,1);
    for p = 1:P
        alpha(p) = expected_beta(p,:)*(sum_expected_beta-expected_beta(p,:))';
    end
    gamma.phi_shp = prior.shape.phi+cu_bar;
    gamma.phi_rte = SUMG_User_Product*alpha+gamma.psi_shp./gamma.psi_rte;
end

%% psi
function [gamma] = updatePsi(prior,gamma)
    gamma.psi_shp = prior.shape.psi+prior.shape.phi;
    gamma.psi_rte = prior.rate.psi+gamma.phi_shp./gamma.phi_rte;
end