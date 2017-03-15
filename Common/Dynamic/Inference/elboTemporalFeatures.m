function res = elboTemporalFeatures(U,P,K,I,J,T,inedges,B,...
        sum_sij_gw,sum_entropy_multinomial,...
        fixed_shape,fixed_rate,gamma,...
        cuik,cpjk,cuv,FijT,Fuij)
    %% initialization
    expected_beta = gamma.beta_shp ./ gamma.beta_rte;
    expected_log_beta = (psi(gamma.beta_shp) - log(gamma.beta_rte));    
    expected_theta = gamma.theta_shp ./ gamma.theta_rte;
    expected_log_theta =(psi(gamma.theta_shp)- log(gamma.theta_rte));
    expected_tau = zeros(U,U);
    expected_log_tau = zeros(U,U);
    for v=1:U
        for u=inedges{v}
            expected_tau(u,v) = gamma.tau_shp(u,v)/gamma.tau_rte(u,v);
            expected_log_tau(u,v) = psi(gamma.tau_shp(u,v))-log(gamma.tau_rte(u,v));
        end
     end
    expected_ksi = gamma.ksi_shp ./ gamma.ksi_rte;
    expected_log_ksi = (psi(gamma.ksi_shp) - log(gamma.ksi_rte));
    expected_eta  = gamma.eta_shp  ./ gamma.eta_rte;
    expected_log_eta  = (psi(gamma.eta_shp)  - log(gamma.eta_rte));
    expected_mu  = gamma.mu_shp  ./ gamma.mu_rte;
    expected_log_mu  = (psi(gamma.mu_shp)  - log(gamma.mu_rte));
    
    res = 0.0;
    %% E[ln p(theta | eta)]
    for u=1:U
        for i=1:I
            for k=1:K
                res = res...
                    +fixed_shape.theta*expected_log_eta(u)...
                    +(fixed_shape.theta-1)*expected_log_theta(u,i,k)...
                    -expected_eta(u)*expected_theta(u,i,k);
            end
        end
    end
    %% E[ln p(beta | ksi)]
    for p=1:P
        for j=1:J
            for k=1:K
                res = res...
                    +fixed_shape.beta*expected_log_ksi(p)...
                    +(fixed_shape.beta-1)*expected_log_beta(p,j,k)...
                    -expected_ksi(p)*expected_beta(p,j,k);
            end
        end
    end
    
    %% E[ln p(tau | mu)]
    for v=1:U
        for u=inedges{v}
            res = res+...
                 fixed_shape.tau*expected_log_mu(u)...
                +(fixed_shape.tau-1)*expected_log_tau(u,v)...
                -expected_mu(u)*expected_tau(u,v); 
        end
    end
    %% E[ln (ksi)]
    for p=1:P
        res = res+...
            (fixed_shape.ksi-1)*expected_log_ksi(p)...
            -fixed_rate.ksi*expected_ksi(p);
    end
    %% E[ln (eta)]
    for u=1:U
        res = res+...
            (fixed_shape.eta-1)*expected_log_eta(u)...
            -fixed_rate.eta*expected_eta(u);
    end
    %% E[ln (mu)]
    for u=1:U
        res = res+...
            (fixed_shape.mu-1)*expected_log_mu(u)...
            -fixed_rate.mu*expected_mu(u);
    end
    %% E[ln p(E,S|theta,beta,tau)]
    %cupijk * ln theta(u,i,k)
    for u=1:U
        for i=1:I
            for k=1:K
                res = res...
                    +cuik(u,i,k)*expected_log_theta(u,i,k);
            end
        end
    end
    %cupijk * ln beta(p,j,k)
    for p=1:P
        for j=1:J
            for k=1:K
                res = res...
                    +cpjk(p,j,k)*expected_log_beta(p,j,k);
            end
        end
    end
    %cuv
    for u=1:U
        for v=inedges{u}
            %disp(['v=' , num2str(v) , ' u=' , num2str(u) , ' cuv=' , num2str(cuv(v,u)) , 'expected_log_tau =', num2str(expected_log_tau(v,u))]);
            res = res+cuv(v,u)*expected_log_tau(v,u);
        end
    end
    
    res = res+sum_sij_gw;
    
   %% sigma(theta(u,i)*beta(p,i)*F(i,j,T))
   %{
    sumTheta = zeros(1,I,K);
    for u=1:U
        sumTheta = sumTheta+expected_theta(u,:,:);
    end
%}
    sumBeta = zeros(1,J,K);
    for p=1:P
        sumBeta = sumBeta+expected_beta(p,:,:);
    end
    %{
    for u=1:U
        for i=1:I
            for j=1:J
                currentTheta = expected_theta(u,i,:);
                currentTheta = currentTheta(:);
                
                %we use sum of beta instead of looping through products to speed up.
                currentBeta =  sumBeta(1,j,:);
                currentBeta = currentBeta(:);
                %disp(currentTheta);
                %disp(currentBeta);
                res = res-currentTheta'*currentBeta*(FijT(i,j)-Fuij(u,i,j)); %before formula (5) : sigma(Theta(u,i)*Beta(p,j)*F(i,j,T))
            end
        end
    end
    %}
    for i=1:I
        for j=1:J
            x = squeeze(expected_theta(:,i,:))*squeeze(sumBeta(1,j,:)); %x is U*1
            res = res-sum(x)*FijT(i,j)+sum(x.*Fuij(:,i,j));
        end
    end
    
    %% sigma (tau(v,u) * G(T-ti))
    for v=1:U
        for u=inedges{v}
            res = res - expected_tau(u,v)*B(u,v);
        end
    end
    %% ENTROPY
    res = res+sum(sum(sum(gamma_entropy(gamma.beta_shp ,gamma.beta_rte))));
    res = res+sum(sum(sum(gamma_entropy(gamma.theta_shp,gamma.theta_rte))));
    for v=1:U
        for u=inedges{v}
            res = res+gamma_entropy(gamma.tau_shp(u,v),gamma.tau_rte(u,v));
        end
    end
    res = res+sum(gamma_entropy(gamma.ksi_shp , gamma.ksi_rte));
    res = res+sum(gamma_entropy(gamma.eta_shp , gamma.eta_rte));
    res = res+sum(gamma_entropy(gamma.mu_shp  , gamma.mu_rte));
    res = res+sum_entropy_multinomial;
end

function entropy = gamma_entropy(shp , rte)
    entropy = shp - log(rte) + gammaln(shp) + (1-shp).*psi(shp);
end