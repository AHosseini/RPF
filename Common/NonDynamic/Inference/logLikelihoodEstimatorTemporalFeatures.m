function lle = logLikelihoodEstimatorTemporalFeatures(inedges,...
    eventsMatrix,tauMatrix,thetaMatrix,betaMatrix, B, params, ...
    w, g, g_factorized, T,t0)

U = params.U;
P = params.P;

tauSparse = cell(U,1);
for u=1:U
    tauSparse{u} = sparse(1,U);
    for i=1:length(inedges{u})
        v = inedges{u}(i);
        tauSparse{u}(v) = tauMatrix(v,u);
    end
end
    
%now compute lle
lle = 0.0;

%% log(lambda(t))
for u=1:U
    for p=1:P
        ptr = ones(U,1);
        currentSocialIntensity = 0.0;
        for i=1:length(eventsMatrix{u,p})
            ti = eventsMatrix{u,p}(i);
            %% theta,beta
            currentTheta = thetaMatrix(u,:)';
            currentBeta = betaMatrix(p,:)';
            %% social intensity
            if (i > 1)
                currentSocialIntensity = currentSocialIntensity*...
                    g_factorized(ti-eventsMatrix{u,p}(i-1),w,1);
            end
            for v=inedges{u}
                while(ptr(v) <= length(eventsMatrix{v,p}))
                    tj = eventsMatrix{v,p}(ptr(v));
                    if (tj >= ti) %only events before i can influence
                        break;
                    end
                    currentSocialIntensity = currentSocialIntensity+tauMatrix(v,u)*g(ti-tj,w);
                    ptr(v) = ptr(v)+1;
                end
            end
            %% sum of social intensity and base intensity
            lle = lle+log(currentTheta'*currentBeta+currentSocialIntensity);
        end
    end
end

%%% sigma(theta(u,i)*beta(p,i)*F(i,j,T))
sumTheta = (T-t0)'*thetaMatrix; % (1*U) * (U*K) = 1*K
sumBeta = sum(betaMatrix);
lle = lle-sumTheta*sumBeta';
%% sigma(tau(u,v) G(w,T-ti))
for u=1:U
    lle = lle-sum(tauMatrix(inedges{u},u).*B(inedges{u},u));
end
end