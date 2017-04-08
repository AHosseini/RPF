function [theta,beta,tau,gamma,prior,cuv] = NonDynamicRPF(events,eventsMatrix,T,t0,inedges,outedges,prior,params,kernel)
U = params.U;
P = params.P;
K = params.K;
datasetName = params.datasetName;
methodName = params.methodName;
maxNumberOfIterations = params.maxNumberOfIterations;
saveInterval = params.saveInterval;
plottingInIteration = params.plottingInIteration;
if nargin<9
    w=1;
    g = @(x,w) w*exp(-w*x);
    g_factorized = @(x,w,p) p*exp(-w*x);
    g_log = @(x,w) log(w)-w*x;
    G = @(x,w) 1-exp(-w*x);
else
    w = kernel.w;
    g = kernel.g;
    g_factorized = kernel.g_factorized;
    G = kernel.G;
    g_log = kernel.g_log;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%       compute B Values          %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%    B(u)= \sum_{e\in H_u} G(T-t_e) %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
B = zeros(U,U);
for u=1:U
    for p=1:P
        for i=1:length(eventsMatrix{u,p})
            te = eventsMatrix{u,p}(i);
            for v=outedges{u}
                B(u,v) = B(u,v)+ G(T-te,w);
                if (t0(v) > te)
                    B(u,v) = B(u,v) - G(t0(v)-te,w);
                end
            end            
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma = initializeGamma(inedges, outedges, prior,params);
iteration = 0;

x = 1:maxNumberOfIterations;
elboList = zeros(maxNumberOfIterations,1);
logLikelihoodList = zeros(maxNumberOfIterations,1);
oldLogLikelihood = -inf;
[theta, beta, tau] = estimateParams(gamma,inedges,U);

newLogLikelihood = logLikelihoodEstimatorTemporalFeatures(inedges,...
                    eventsMatrix,tau,theta,beta, B, params, ...
                    w, g, g_factorized, T,t0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (newLogLikelihood-oldLogLikelihood > 1e-5 && iteration<maxNumberOfIterations)    
    iteration = iteration+1;
    fprintf('Iteration %d\n',iteration)
    if (strcmp(params.phiComputationAlgorithm ,'quadratic'))
        tic
        [cuv, cuk, cpk,sum_entropy_multinomial,sum_sij_gw] = ...
        quadraticTimeComputePhi(eventsMatrix, events, inedges, gamma, w, g_log, params);
        fprintf('1-QuadraticTimeComputePhi Completed.Elapsed Time is %f\n',toc);
    elseif (strcmp(params.phiComputationAlgorithm,'linear'))
        [cuv,cuk,cpk] = linearTimeComputePhi(...
            U,P,K,inedges,eventsMatrix,events,...
            gamma,w,g,g_log,g_factorized);
    else
        disp('no valid phi algorithm specified, phi algorithm should be "linear" or "quadratic"');
        break;
    end
    tic
    [gamma] = updateTau(U,inedges,prior,cuv,B,gamma);   
    fprintf('2-Tau Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updateTheta(U,K,cuk,prior,gamma, T,t0);
    fprintf('3-Theta Parameters Updated.Elapsed Time is %f\n',toc);
    tic;
    [gamma] = updateBeta(P,K,cpk,prior,gamma, T,t0);
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
%%%%%%%%%%%% Saving Elbo and LogLikelihood %%%%%%%%%%%%%%%%%
    if (strcmp(params.phiComputationAlgorithm,'quadratic'))
        tic;
        elboList(iteration) = ELBO(U,P,K,T,inedges,B,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior,gamma,...
                cuk,cpk,cuv,t0);
        fprintf('Elbo Computed. Elapsed Time is %f\n',toc);
    end
    
    % LogLikelihood
    tic;
    oldLogLikelihood = newLogLikelihood;
    
    [theta, beta, tau] = estimateParams(gamma,inedges,U);
     newLogLikelihood = logLikelihoodEstimatorTemporalFeatures(inedges,...
                        eventsMatrix,tau,theta,beta, B, params, ...
                        w, g, g_factorized, T,t0);
    fprintf('LogLikelihood Computed.Elapsed Time is %f\n',toc);
    
    logLikelihoodList(iteration) = newLogLikelihood;
    
    if iteration>1 && strcmp(params.phiComputationAlgorithm,'quadratic')
        fprintf('%d\t DeltaElbo=%f\t deltaLL=%f. \n',iteration,elboList(iteration)-elboList(iteration-1),...
            logLikelihoodList(iteration)-logLikelihoodList(iteration-1));
    elseif iteration > 1
        fprintf('%d\t deltaLL=%f. \n',iteration,logLikelihoodList(iteration)-logLikelihoodList(iteration-1));
    end
    
    if mod(iteration,saveInterval) ==0
       save(sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,iteration),...
        'theta','beta','tau','kernel','params','prior','gamma','cuv');
    end
    
%%  Plotting Results
    if (plottingInIteration == 1)
        subplot(2,1,1);
        plot(x(1:plotDataCount),elboList(1:plotDataCount));
        subplot(2,1,2);
        plot(x(1:plotDataCount),logLikelihoodList(1:plotDataCount));
        drawnow
    end
end
save(sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,maxNumberOfIterations),...
    'theta','beta','tau','kernel','params','prior','gamma','cuv');
[theta, beta, tau] = estimateParams(gamma,inedges,U);
end

function [theta, beta, tau] = estimateParams(gamma,inedges,U)
    theta = gamma.theta_shp ./ gamma.theta_rte;
    beta = gamma.beta_shp ./ gamma.beta_rte;
    tau = zeros(U,U);
    for v=1:U
        tau(inedges{v},v) = gamma.tau_shp(inedges{v},v) ./ gamma.tau_rte(inedges{v},v);  
    end
end
%% 2.2 theta
function [gamma] = updateTheta(U,K,cuk,prior,gamma, T,t0)
    sum_expected_beta_p = squeeze(sum(gamma.beta_shp./gamma.beta_rte,1)); % 1*K matrix
    gamma.theta_shp = prior.shape.theta+cuk;
    gamma.theta_rte = (T-t0) * sum_expected_beta_p +repmat(gamma.eta_shp./gamma.eta_rte,[1,K]);
end
%% 2.3 beta
function [gamma] = updateBeta(P,K,cpk,prior,gamma, T,t0)
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
        user_expected_tau = sum(gamma.tau_shp(u,outedges{u})/gamma.tau_rte(u,outedges{u}));
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
