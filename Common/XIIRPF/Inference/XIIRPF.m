function [expected_theta,expected_beta,expected_tau,expected_phi, ...
    expectedPi, gamma,prior,cuv,cbb] =...
    XCoRPF(events, eventsMatrix,userEvents, itemsCluster,itemsSimilarity,...
        T, t0, inedges, outedges, prior, params, kernel, likelihoodIsRequired)
U = params.U;
P = params.P;
K = params.K;
B = params.B;
datasetName = params.datasetName;
methodName = params.methodName;
maxNumberOfIterations = params.maxNumberOfIterations;
startIteration = params.startIteration;
saveInterval = params.saveInterval;
plottingInIteration = params.plottingInIteration;
if nargin<16
    likelihoodIsRequired = 0;
end
if nargin<15
    kernel = struct;
    kernel.w1 = 0.1;
    kernel.w2 = 0.1;
    kernel.g = @(x,w) exp(-w*x);
    kernel.g_log = @(x,w) -w*x;
    kernel.G = @(x,w) 1/w*(1-exp(-w*x));

    kernel.nu = 1;
    kernel.d = @(dist) (dist)/kernel.nu;
    kernel.d_log = @(dist) log(dist)-log(kernel.nu);
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
                SUMG_User(u,v) = SUMG_User(u,v)+ kernel.G(T-te,kernel.w1);
                if (t0(v) > te)
                    SUMG_User(u,v) = SUMG_User(u,v) - kernel.G(t0(v)-te,kernel.w1);
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
    SUMG_User_Product(un,pn) =  SUMG_User_Product(un,pn)+kernel.G(T-tn,kernel.w2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if startIteration ==0
    rng(0);
    gamma = initializeGamma(inedges, outedges, prior,params);
    iteration = 0;
else
    iteration = startIteration;
    modelFileName = sprintf('../../../../LearnedModels/LearnedModel_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',...
            methodName,datasetName,params.K,kernel.w1,kernel.w2,iteration);
    load(modelFileName);
end

plotDataCount = maxNumberOfIterations;  
x = 1:maxNumberOfIterations;
elboList = zeros(maxNumberOfIterations,1);
logLikelihoodList = zeros(maxNumberOfIterations,1);
% oldLogLikelihood = -inf;
if likelihoodIsRequired
    [expected_theta,  expected_beta, expected_tau, expected_phi, expectedPi] = ...
    estimateParams(gamma,inedges,U);

    likelihood = logLikelihoodEstimator(events,itemsCluster,itemsSimilarity, outedges, ...
                        expected_tau, expected_theta, expected_beta,...
                        expected_phi, expectedPi, SUMG_User, params, ...
                        kernel, T,t0);
    fprintf('loglikelihood is %.2f\n',likelihood);
end

% diffElboS= zeros(maxNumberOfIterations,1);
% diffElboTheta = zeros(maxNumberOfIterations,1);
% diffElboBeta = zeros(maxNumberOfIterations,1);
% diffElboTau = zeros(maxNumberOfIterations,1);
% diffElboPhi = zeros(maxNumberOfIterations,1);
% diffElboPi = zeros(maxNumberOfIterations,1);
% diffElboKsi = zeros(maxNumberOfIterations,1);
% diffElboMu = zeros(maxNumberOfIterations,1);
% diffElboEta = zeros(maxNumberOfIterations,1);
% diffElboPsi= zeros(maxNumberOfIterations,1);
% diffElboRho= zeros(maxNumberOfIterations,1);

elbo = -inf;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (iteration<maxNumberOfIterations)% && newLogLikelihood-oldLogLikelihood > 1e-5)
    iteration = iteration+1;
    fprintf('Iteration %d\n',iteration)
    tic
    [cuv, cuk, cpk,cbb,cu_bar,sum_entropy_multinomial,sum_sij_gw,sum_sij_d] = ...
    quadraticTimeComputeS(events,userEvents,itemsCluster,itemsSimilarity, ...
        gamma, kernel, params);
    fprintf('1-S parameters updated. Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboS(iteration) = elbo-lastElbo;
%     if diffElboS(iteration)<-1e-5
%         fprintf('BUG: S-DiffElbo is negative:%.2f\n',diffElboS(iteration));
% %         exit()
%     end
%     fprintf('%d-S-DiffElbo is:%.2f\n',iteration,diffElboS(iteration));
    tic
    [gamma] = updateTau(U,inedges,prior,cuv,SUMG_User,gamma);   
    fprintf('2-Tau Parameters Updated. Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboTau(iteration) = elbo-lastElbo;  
%     if diffElboTau(iteration)<-1e-5
%         fprintf('BUG: Tau-DiffElbo is negative:%.2f\n',diffElboTau(iteration));
% %         exit()
%     end
%     fprintf('%d-Tau-DiffElbo is:%.2f\n',iteration,diffElboTau(iteration));
    tic;
    [gamma] = updateTheta(K,cuk,prior,gamma, T,t0);
    fprintf('3-Theta Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
% 
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboTheta(iteration) = elbo-lastElbo;
%     if diffElboTheta(iteration)<-1e-5
%         fprintf('BUG: Theta-DiffElbo is negative:%.2f\n',diffElboTheta(iteration));
% %         exit()
%     end
%     fprintf('%d-Theta-DiffElbo is:%.2f\n',iteration,diffElboTheta(iteration));
    tic;
    [gamma] = updateBeta(P,K,cpk,prior,gamma, T,t0);
    fprintf('4-Beta Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboBeta(iteration) = elbo-lastElbo;  
%     if diffElboBeta(iteration)<-1e-5
%         fprintf('BUG: Beta-DiffElbo is negative:%.2f\n',diffElboBeta(iteration));
% %         exit()
%     end
%     fprintf('%d-Beta-DiffElbo is:%.2f\n',iteration,diffElboBeta(iteration));
    tic;
    [gamma] = updateEta(U,prior,gamma);
    fprintf('5-Eta Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboEta(iteration) = elbo-lastElbo;  
%     if diffElboEta(iteration)<-1e-5
%         fprintf('BUG: Eta-DiffElbo is negative:%.2f\n',diffElboEta(iteration));
% %         exit()
%     end
%     fprintf('%d-Eta-DiffElbo is:%.2f\n',iteration,diffElboEta(iteration));
    tic;
    [gamma] = updateMu(U,outedges,prior,gamma);
    fprintf('6-Mu Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboMu(iteration) = elbo-lastElbo;  
%     if diffElboMu(iteration)<-1e-5
%         fprintf('BUG: Mu-DiffElbo is negative:%.2f\n',diffElboMu(iteration));
% %         exit()
%     end
%     fprintf('%d-Mu-DiffElbo is:%.2f\n',iteration,diffElboMu(iteration));
    tic;
    [gamma] = updateKsi(P,prior,gamma);
    fprintf('7-Ksi Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboKsi(iteration) = elbo-lastElbo;  
%     if diffElboKsi(iteration)<-1e-5
%         fprintf('BUG: Ksi-DiffElbo is negative:%.2f\n',diffElboKsi(iteration));
% %         exit()
%     end
%     fprintf('%d-Ksi-DiffElbo is:%.2f\n',iteration,diffElboKsi(iteration));
    
    tic;
    [gamma] = updatePsi(prior,gamma);
    fprintf('8-Psi Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboPsi(iteration) = elbo-lastElbo;
%     if diffElboPsi(iteration)<-1e-5
%         fprintf('BUG: Psi-DiffElbo is negative:%.2f\n',diffElboPsi(iteration));
% %         exit()
%     end
%     fprintf('%d-Psi-DiffElbo is:%.2f\n',iteration,diffElboPsi(iteration));

    tic;
    [gamma] = updatePhi(P, itemsCluster, itemsSimilarity, prior,gamma, kernel, cu_bar ,SUMG_User_Product);
    fprintf('9-Phi Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboPhi(iteration) = elbo-lastElbo;  
%     if diffElboPhi(iteration)<-1e-5
%         fprintf('BUG: Phi-DiffElbo is negative:%.2f\n',diffElboPhi(iteration));
% %         exit()
%     end
%     fprintf('%d-Phi-DiffElbo is:%.2f\n',iteration,diffElboPhi(iteration));
    
    tic;
    [gamma] = updatePi(B,cbb, itemsCluster, itemsSimilarity, prior,gamma, kernel ,SUMG_User_Product);
    fprintf('10-Pi Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboPi(iteration) = elbo-lastElbo;  
%     if diffElboPi(iteration)<-1e-5
%         fprintf('BUG: Pi-DiffElbo is negative:%.2f\n',diffElboPi(iteration));
% %         exit()
%     end
%     fprintf('%d-Pi-DiffElbo is:%.2f\n',iteration,diffElboPi(iteration));
    
    tic;
    [gamma] = updateRho(B, prior,gamma);
    fprintf('11-rho Parameters Updated.Elapsed Time is %f\n',toc);
%     lastElbo = elbo;
%     elbo =ELBO(T, itemsCluster, itemsSimilarity,...
%             inedges,SUMG_User,SUMG_User_Product,...
%             sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
%             prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
%     diffElboRho(iteration) = elbo-lastElbo;
%     if diffElboRho(iteration)<-1e-5
%         fprintf('BUG: Rho-DiffElbo is negative:%.2f\n',diffElboRho(iteration));
% %         exit()
%     end
%     fprintf('%d-Rho-DiffElbo is:%.2f\n',iteration,diffElboRho(iteration));

%%%%%%%%%%%% Saving Elbo and LogLikelihood %%%%%%%%%%%%%%%%%
    tic;
    elboList(iteration)=ELBO(T, itemsCluster, itemsSimilarity,...
            inedges,SUMG_User,SUMG_User_Product,...
            sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
            prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params);
    fprintf('Elbo Computed. Elapsed Time is %f\n',toc);
    if likelihoodIsRequired
        tic;
        [expected_theta,  expected_beta, expected_tau, expected_phi, expectedPi] = ...
            estimateParams(gamma,inedges,U);
        likelihood = logLikelihoodEstimator(events,itemsCluster,itemsSimilarity, outedges, ...
                        expected_tau, expected_theta, expected_beta,...
                        expected_phi, expectedPi, SUMG_User, params, ...
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
        [expected_theta,  expected_beta, expected_tau, expected_phi, expectedPi] = ...
            estimateParams(gamma,inedges,U);
        modelFileName = sprintf('../../../../LearnedModels/LearnedModel_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',...
            methodName,datasetName,params.K,kernel.w1,kernel.w2,iteration);
       save(modelFileName,...
        'expected_theta','expected_beta','expected_tau','expected_phi','expectedPi','kernel','params','prior','gamma','cuv','cbb');
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
[expected_theta,  expected_beta, expected_tau, expected_phi, expectedPi] = ...
            estimateParams(gamma,inedges,U);
        modelFileName = sprintf('../../../../LearnedModels/LearnedModel_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',...
            methodName,datasetName,params.K,kernel.w1,kernel.w2,maxNumberOfIterations);
save(modelFileName,...
    'expected_theta','expected_beta','expected_tau','expected_phi','expectedPi','kernel','params','prior','gamma','cuv','cbb');
end

function [expected_theta,  expected_beta, ...
    expected_tau, expected_phi, expected_pi] = estimateParams(gamma,inedges,U)
    expected_beta = gamma.beta_shp ./ gamma.beta_rte;
    expected_theta = gamma.theta_shp ./ gamma.theta_rte;
    expected_tau = zeros(U,U);
    for v=1:U
        expected_tau(inedges{v},v) = gamma.tau_shp(inedges{v},v)./gamma.tau_rte(inedges{v},v);
    end
    expected_phi = gamma.phi_shp ./ gamma.phi_rte;
    expected_pi = gamma.pi_shp ./ gamma.pi_rte;
end
%% 2.2 theta
function [gamma] = updateTheta(K,cuk,prior,gamma, T,t0)
    sum_expected_beta_p = squeeze(sum(gamma.beta_shp./gamma.beta_rte,1)); % 1*K matrix
    gamma.theta_shp = prior.shape.theta+cuk;
%    gamma.theta_rte = T * repmat(sum_expected_beta_p,[U,1])+repmat(gamma.eta_shp./gamma.eta_rte,[1,K]);
    %(T-t0) * sum_expected_beta_p : (U*1) * (1*K) = U*K
    gamma.theta_rte = (T-t0) * sum_expected_beta_p +repmat(gamma.eta_shp./gamma.eta_rte,[1,K]);
end
%% 2.3 beta
function gamma = updateBeta(P,K,cpk,prior,gamma, T,t0)
    sum_expected_theta_u = (T-t0)'*(gamma.theta_shp./gamma.theta_rte); %(1*U) * (U*K) = 1*K
    gamma.beta_shp = prior.shape.beta+cpk;
    gamma.beta_rte = repmat(sum_expected_theta_u,[P,1])+repmat(gamma.ksi_shp./gamma.ksi_rte,[1,K]);

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
function [gamma] = updatePhi(P, itemsCluster, itemDistances, prior,gamma, kernel, cu_bar ,SUMG_User_Product)
    expected_pi = gamma.pi_shp./gamma.pi_rte;
    alpha = zeros(P,1);
    for p = 1:P
        alpha(p) = expected_pi(itemsCluster(p),itemsCluster(1:P))*kernel.d(itemDistances(p,:))'...
            -expected_pi(itemsCluster(p),itemsCluster(p))*kernel.d(itemDistances(p,p))';    
    end
    gamma.phi_shp = prior.shape.phi+cu_bar;
    gamma.phi_rte = SUMG_User_Product*alpha+gamma.psi_shp./gamma.psi_rte;
end

%% psi
function [gamma] = updatePsi(prior,gamma)
    gamma.psi_shp = prior.shape.psi+prior.shape.phi;
    gamma.psi_rte = prior.rate.psi+gamma.phi_shp./gamma.phi_rte;
end
%% pi
function [gamma] = updatePi(B,cbb, itemsCluster, itemDistances, prior,gamma, kernel ,SUMG_User_Product)
    gamma.pi_shp = prior.shape.pi+cbb;
    expected_rho = gamma.rho_shp./gamma.rho_rte;
    expected_phi = gamma.phi_shp./gamma.phi_rte;
    for b = 1:B
        for c = 1:B
            dist = sum(kernel.d(itemDistances(itemsCluster==b,itemsCluster==c)),2);
            gamma.pi_rte(b,c) = (SUMG_User_Product(:,itemsCluster==b)*dist)'*expected_phi+expected_rho(b);
        end
    end
end
%% rho
function [gamma] = updateRho(B, prior,gamma)
    expected_pi = gamma.pi_shp./gamma.pi_rte;
    gamma.rho_shp = prior.shape.rho+B*prior.shape.pi;
    gamma.rho_rte = prior.rate.rho+sum(expected_pi,2);
end