function lle = logLikelihoodEstimatorTemporalFeatures(inedges,...
    eventsMatrix,tauMatrix,thetaMatrix,betaMatrix, B, FijT,Fuij, params, ...
    w, g, g_factorized)

U = params.U;
P = params.P;
K = params.K;
I = params.I;
J = params.J;

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
% errorNewSocial = zeros(U,P);
for u=1:U
    for p=1:P
        ptr = ones(U,1);
        currentSocialIntensity = 0.0;
        for i=1:length(eventsMatrix{u,p})
            ti = eventsMatrix{u,p}(i);
            %% theta,beta
            [day,hour] = dayAndHour(ti);
            currentTheta = thetaMatrix(u,day+1,:)+thetaMatrix(u,hour+8,:);
            currentTheta = currentTheta(:);
            currentBeta = betaMatrix(p,day+1,:)+betaMatrix(p,hour+8,:);
            currentBeta = currentBeta(:);
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

%             errorNewSocial(u,p) = abs(currentSocialIntensity-...
%                             socialIntensity(tauSparse,eventsMatrix,ti,w,g,u,p) )^2;
        end
    end
end

%% sigma(theta(u,i)*beta(p,i)*F(i,j,T))
%{
sumTheta = zeros(1,I,K);
for u=1:U
    sumTheta = sumTheta+thetaMatrix(u,:,:);
end
%}
sumBeta = zeros(1,J,K);
for p=1:P
    sumBeta = sumBeta+betaMatrix(p,:,:);
end

%{
for u=1:U
    for i=1:I
        for j=1:J
            currentTheta = thetaMatrix(u,i,:);
            currentTheta = currentTheta(:);
            
            %we use sum of beta instead of looping through products to speed up.
            currentBeta =  sumBeta(1,j,:); 
            currentBeta = currentBeta(:);

            lle = lle-currentTheta'*currentBeta*(FijT(i,j)-Fuij(u,i,j)); %before formula (5) : sigma(Theta(u,i)*Beta(p,j)*F(i,j,T))
        end
    end
end
%}
for i=1:I
    for j=1:J
        x = squeeze(thetaMatrix(:,i,:))*squeeze(sumBeta(1,j,:)); %x is U*1
        lle = lle-sum(x)*FijT(i,j)+sum(x.*Fuij(:,i,j));
    end
end

%% sigma(tau(u,v) G(w,T-ti))
for u=1:U
    for i=1:length(inedges{u})
        v = inedges{u}(i);
        lle = lle-tauMatrix(v,u)*B(v,u);
    end
end

end
