function lle = logLikelihoodEstimator(events, itemsCluster, itemDistances,...
    outedges,tauMatrix, thetaMatrix, betaMatrix, phi, pi, SUMG_User, params, ...
    kernel, T, t0)

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
        alphaMatrix(p,q) = pi(itemsCluster(p),itemsCluster(q))*kernel.d(itemDistances(p,q));
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
        socialIntensity(ui,pi) * kernel.g(ti-socialIntensityLastUpdate(ui,pi),kernel.w1);
    interItemIntensity(ui,pi) = interItemIntensity(ui,pi)* ...
        kernel.g(ti-interItemIntensityLastUpdate(ui,pi),kernel.w2);
    intensity = baseIntensity + socialIntensity(ui,pi) + phi(ui)*interItemIntensity(ui,pi);
    lle = lle+log(intensity)-phi(ui)*sum(alphaMatrix(pi,:))*kernel.G(T-ti,kernel.w2);
    socialIntensity(outedges{ui},pi) = socialIntensity(outedges{ui},pi)...
        .*kernel.g(ti-socialIntensityLastUpdate(outedges{ui},pi),kernel.w1)+...
        tauMatrix(ui,outedges{ui})';
    socialIntensityLastUpdate(outedges{ui},pi) = ti;
    interItemIntensity(ui,:) = interItemIntensity(ui,:)...
        .*kernel.g(ti-interItemIntensityLastUpdate(ui,:),kernel.w1)+alphaMatrix(pi,:);
    socialIntensityLastUpdate(ui,:) = ti;
end
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