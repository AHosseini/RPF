function [theta,beta,tau,gamma,prior,cuv,maeList,logLikelihoodList] = ...
    DynamicRPF(events,eventsMatrix,T,t0,inedges,outedges,prior,params,kernel,isSynthetic,modelParams)
U = params.U;
P = params.P;
K = params.K;
I = params.I;
J = params.J;
datasetName = params.datasetName;
methodName = params.methodName;
maxNumberOfIterations = params.maxNumberOfIterations;
saveInterval = params.saveInterval;
plottingInIteration = params.plottingInIteration;
if nargin<10
    isSynthetic = 0;
end
if nargin<9
    w=1;
    g = @(x,w) w*exp(-w*x);
    g_factorized = @(x,w,p) p*exp(-w*x);
    g_log = @(x,w) log(w)-w*x;
    G = @(x,w) 1-exp(-w*x) ;
    HL = @(i,j,T) integralOfProduct(i,j,T);
else
    w = kernel.w;
    g = kernel.g;
    g_factorized = kernel.g_factorized;
    G = kernel.G;
    HL = kernel.F;
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

FijT = zeros(I,J);
for i = 1:I
    for j = 1:J
        FijT(i,j) = HL(i,j,T);
    end
end

Fuij = zeros(U,I,J);
for u=1:U
    for i = 1:I
        for j = 1:J
            Fuij(u,i,j) = HL(i,j,t0(u));
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Initialization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gamma = initializeGamma(inedges, outedges, prior,params);
iteration = 0;
plotDataCount = 0;

x = 1:maxNumberOfIterations;
elboList = zeros(maxNumberOfIterations,1);
logLikelihoodList = zeros(maxNumberOfIterations,1);
maeList = zeros(maxNumberOfIterations,1);
mseList = zeros(maxNumberOfIterations,1);
mreList = zeros(maxNumberOfIterations,1);
oldLogLikelihood = -inf;
[theta, beta, tau] = estimateParams(gamma,inedges,U);

newLogLikelihood = logLikelihoodEstimatorTemporalFeatures(inedges,...
                eventsMatrix,tau,theta,beta, B, FijT,Fuij, params, ...
                w, g, g_factorized);

newElbo = -inf;
lastElbo = newElbo;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Main Loop %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while (newLogLikelihood-oldLogLikelihood > 1e-5 && iteration<maxNumberOfIterations)
    iteration = iteration+1;
    fprintf('Iteration %d\n',iteration)
    if (strcmp(params.phiComputationAlgorithm ,'quadratic'))
        tic
        [cuv, cuik, cpjk,sum_entropy_multinomial,sum_sij_gw] = ...
            quadraticTimeComputePhi(eventsMatrix, events, inedges, gamma, w, g, g_log, params);
        fprintf('1-QuadraticTimeComputePhi Completed.Elapsed Time is %f\n',toc);
    elseif (strcmp(params.phiComputationAlgorithm,'linear'))
        tic    
        [cuv, cuik, cpjk] = linearTimeComputePhi(...
                U,P,K,I,J,inedges,eventsMatrix,events,...
                gamma,w,g,g_factorized);
        fprintf('1-LinearTimeComputePhi Completed.Elapsed Time is %f\n',toc);
    else
        disp('no valid phi algorithm specified, phi algorithm should be "linear" or "quadratic"');
        break;
    end
   
        
        tic
    [gamma] = updateTau(U,inedges,prior,cuv,B,gamma);   
    fprintf('2-Tau Parameters Updated.Elapsed Time is %f\n',toc);
    
    tic;
    [gamma] = updateTheta(U,K,I,J,cuik,prior,gamma,FijT,Fuij);
    fprintf('3-Theta Parameters Updated.Elapsed Time is %f\n',toc);
    
    tic;
    [gamma] = updateBeta(U,P,K,I,J,cpjk,prior,gamma,FijT,Fuij);
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
    %Elbo
    if (strcmp(params.phiComputationAlgorithm,'quadratic'))
        lastElbo = newElbo;
        tic;
        newElbo = elboTemporalFeatures(U,P,K,I,J,T,inedges,B,...
                sum_sij_gw,sum_entropy_multinomial,...
                prior.shape,prior.rate,gamma,...
                cuik,cpjk,cuv,FijT,Fuij);
        fprintf('Elbo Computed.Elapsed Time is %f\n',toc);
        elboList(iteration) = newElbo;
    end
    
    % LogLikelihood
    
        [theta, beta, tau] = estimateParams(gamma,inedges,U);
    if (mod(iteration,10) == 1)
        oldLogLikelihood = newLogLikelihood;
        tic;
         newLogLikelihood = logLikelihoodEstimatorTemporalFeatures(inedges,...
                            eventsMatrix,tau,theta,beta, B, FijT,Fuij, params, ...
                            w, g, g_factorized);
        fprintf('LogLikelihood Computed.Elapsed Time is %f\n',toc);
        logLikelihoodList(iteration) = newLogLikelihood;
        if iteration>1 && strcmp(params.phiComputationAlgorithm,'quadratic')
            fprintf('iteration %d:\t deltaElbo = %f\t deltaLL=%f. \n',iteration,newElbo-lastElbo,newLogLikelihood-oldLogLikelihood);
            if (newElbo-lastElbo < -0.1)               
                break;
            end        
        else
            fprintf('iteration %d:\t deltaLL=%f. \n',iteration,newLogLikelihood-oldLogLikelihood);
        end
    end
    

    if (isSynthetic)
        numOfElements = numel(theta)+numel(beta)+numel(find(tau > 0));
        maeList(iteration) = (...
            sum(sum(sum(abs(theta-modelParams.theta))))+...
            sum(sum(sum(abs(beta-modelParams.beta))))+....
            sum(sum(abs(tau-modelParams.tau)))...
            ) / numOfElements;
        
        mseList(iteration) = sqrt(...
            (...
             sum(sum(sum((theta-modelParams.theta).^2)))+...
             sum(sum(sum((beta-modelParams.beta).^2)))+...
             sum(sum((tau-modelParams.tau).^2))...
             )/numOfElements ...
            );
        
        tauRelativeError = 0.0;
        for u=1:U
            for v=outedges{u}
                tauRelativeError = tauRelativeError+abs(tau(u,v)-modelParams.tau(u,v))/modelParams.tau(u,v);
            end
        end
        mreList(iteration) =  (...
            sum(sum(sum( abs(theta-modelParams.theta) ./ modelParams.theta)))+...
            sum(sum(sum( abs(beta-modelParams.beta) ./ modelParams.beta )))+...
            tauRelativeError...
            ) / numOfElements;
    end
    
    if mod(iteration,saveInterval) == 1
       save(sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,iteration),...
        'theta','beta','tau','kernel','params','prior','gamma','cuv','mreList','mseList','maeList','logLikelihoodList');
    end
%%  Plotting Results
    if (plottingInIteration == 1)
        subplot(2,1,1);
        plot(x(1:plotDataCount),elboList(1:plotDataCount));
        subplot(2,1,2);
        plot(x(1:plotDataCount),logLikelihoodList(1:plotDataCount));
        drawnow
    end
    disp('==============================================================');
end
fprintf('loop completed with %d iterations\n' , iteration);
save(sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName,datasetName,params.K,w,maxNumberOfIterations),...
        'theta','beta','tau','kernel','params','prior','gamma','cuv','maeList','logLikelihoodList','iteration');
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
function [gamma] = updateTheta(U,K,I,J,cuik,prior,gamma,FijT,Fuij)
    sum_expected_beta_over_time = zeros(U,I,K);
    sum_expected_beta_p = squeeze(sum(gamma.beta_shp./gamma.beta_rte,1)); % J*K matrix
    for u=1:U
        for i = 1:I
            sum_expected_beta_over_time(u,i,:) = sum(repmat(FijT(i,:)'-squeeze(Fuij(u,i,:)),1,K).*sum_expected_beta_p);
        end
    end
    gamma.theta_shp = prior.shape.theta+cuik;
    gamma.theta_rte = sum_expected_beta_over_time+repmat(gamma.eta_shp./gamma.eta_rte,[1,I,K]);

end
%% 2.3 beta
function [gamma] = updateBeta(U,P,K,I,J,cpjk,prior,gamma,FijT,Fuij)
    sum_expected_theta_over_time = zeros(J,K);
    expected_theta_u = gamma.theta_shp./gamma.theta_rte; % U*I*K matrix
    for j = 1:J
        FCoeff = repmat(FijT(:,j)',U,1)-squeeze(Fuij(:,:,j));
        sum_expected_theta_over_time(j,:) = squeeze(sum(sum(repmat(FCoeff,[1,1,K]).*expected_theta_u)))';
    end
    gamma.beta_shp = prior.shape.beta+cpjk;
    gamma.beta_rte = permute(repmat(sum_expected_theta_over_time,[1,1,P]),[3,1,2])+repmat(gamma.ksi_shp./gamma.ksi_rte,[1,J,K]);
end
%% === 2.1 tau
function [gamma] = updateTau(U,inedges,prior,cuv,B,gamma)
    for v=1:U
        gamma.tau_shp(inedges{v},v) = prior.shape.tau+cuv(inedges{v},v);
        gamma.tau_rte(inedges{v},v) = B(inedges{v},v) + gamma.mu_shp(inedges{v})./gamma.mu_rte(inedges{v});
    end
end
%% 2.5 eta
function [gamma] = updateEta(U,prior,gamma)
    for u=1:U
        user_expected_theta =  sum(sum(gamma.theta_shp(u,:,:) ./ gamma.theta_rte(u,:,:)));
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
        product_expected_beta = sum(sum( gamma.beta_shp(p,:,:) ./ gamma.beta_rte(p,:,:)));
        gamma.ksi_rte(p) = prior.rate.ksi + product_expected_beta;
    end
end
