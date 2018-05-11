function lle = logLikelihoodEstimator(events, outedges,...
    tauMatrix,thetaMatrix,betaMatrix, phi,SUMG_User, params, ...
    kernel, T,t0)

U = params.U;
P = params.P;

% tauSparse = cell(U,1);
% for u=1:U
%     tauSparse{u} = sparse(1,U);
%     for i=1:length(outedges{u})
%         v = outedges{u}(i);
%         tauSparse{u}(v) = tauMatrix(u,v);
%     end
% end

alphaMatrix = zeros(P,P);
for p=1:P
    for q=1:P
        if (p == q) 
            continue;
        end
        alphaMatrix(p,q) = betaMatrix(p,:)*betaMatrix(q,:)';
    end
end

%now compute lle
lle = 0.0;

%% log(lambda(t))
% errorNewSocial = zeros(U,P);


socialIntensity = zeros(U,P);
interItemIntensity = zeros(U,P);
socialIntensityLastUpdate = zeros(U,P);
interItemIntensityLastUpdate = zeros(U,P);

for i = 1:length(events)
    ti = events{i}.time;
    ui = events{i}.user;
    pi = events{i}.product;
    baseIntensity = thetaMatrix(ui,:)*betaMatrix(pi,:)';
    socialIntensity(ui,pi) = ...
        socialIntensity(ui,pi) * kernel.g(ti-socialIntensityLastUpdate(ui,pi),kernel.w);
    interItemIntensity(ui,pi) = interItemIntensity(ui,pi)* ...
        kernel.g(ti-interItemIntensityLastUpdate(ui,pi),kernel.w);
    intensity = baseIntensity + socialIntensity(ui,pi) + phi(ui)*interItemIntensity(ui,pi);
    lle = lle+log(intensity)-phi(ui)*sum(alphaMatrix(pi,:))*kernel.G(T-ti,kernel.w);
    socialIntensity(outedges{ui},pi) = socialIntensity(outedges{ui},pi)...
        .*kernel.g(ti-socialIntensityLastUpdate(outedges{ui},pi),kernel.w)+...
        tauMatrix(ui,outedges{ui})';
    socialIntensityLastUpdate(outedges{ui},pi) = ti;
    interItemIntensity(ui,:) = interItemIntensity(ui,:)...
        .*kernel.g(ti-interItemIntensityLastUpdate(ui,:),kernel.w)+alphaMatrix(pi,:);
    socialIntensityLastUpdate(ui,:) = ti;
end

% lastEvent = zeros(U,P);
% 
% 
% for i=1:length(events)
%     ti = events{i}.time;
%     ui = events{i}.user;
%     pi = events{i}.product;
%     %% socialIntensity
%     socialIntensity = sum(...
%         tauMatrix(inedges{ui},ui) .* socialIntensity(inedges{ui},pi) .* g_factorized(ti-lastEvent(inedges{ui},pi),w,1) );
% 
%     %% productIntensity
%     productIntensity =  (socialIntensity(ui,:) .* g_factorized(ti-lastEvent(ui,:),w,1)) * alphaMatrix(:,pi);
%     lle = lle+log(thetaMatrix(ui,:)*betaMatrix(pi,:)'+socialIntensity+productIntensity);
%     %% update    
%     socialIntensity(ui,pi) = socialIntensity(ui,pi)*g_factorized(ti-lastEvent(ui,pi),w,1)+g(0,w);
%     lastEvent(ui,pi) = ti;
% end


%%% sigma(theta(u,i)*beta(p,i)*F(i,j,T))
sumTheta = (T-t0)'*thetaMatrix; % (1*U) * (U*K) = 1*K
sumBeta = sum(betaMatrix);
lle = lle-sumTheta*sumBeta';
%% sigma(tau(u,v) G(w,T-ti))
for u=1:U
    lle = lle-sum(tauMatrix(u,outedges{u}).*SUMG_User(u,outedges{u}));
end
%% sigma(alpha Gw)
% for p=1:P
%     current = betaMatrix(p,:);
%     lle = lle-(current*(sumBeta-current)')*SUMG_Product(p);
% end
end