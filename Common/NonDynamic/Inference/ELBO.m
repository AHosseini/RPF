function res = ELBO(U,P,K,T,inedges,B,...
            sum_sij_gw,sum_entropy_multinomial,...
            prior,gamma,...
            cuk,cpk,cuv,t0)
    %% initialization
    prior.shape = prior.shape;
    expected_beta = gamma.beta_shp ./ gamma.beta_rte;
    expected_log_beta = (psi(gamma.beta_shp) - log(gamma.beta_rte));    
    expected_theta = gamma.theta_shp ./ gamma.theta_rte;
    expected_log_theta =(psi(gamma.theta_shp)- log(gamma.theta_rte));
    expected_tau = zeros(U,U);
    expected_log_tau = zeros(U,U);
    for v=1:U
        expected_tau(inedges{v},v) = gamma.tau_shp(inedges{v},v)./gamma.tau_rte(inedges{v},v);
        expected_log_tau(inedges{v},v) = psi(gamma.tau_shp(inedges{v},v))-log(gamma.tau_rte(inedges{v},v));
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
        for k=1:K
            res = res...
                +prior.shape.theta*expected_log_eta(u)...
                +(prior.shape.theta-1)*expected_log_theta(u,k)...
                -expected_eta(u)*expected_theta(u,k);
        end
    end
    %% E[ln p(beta | ksi)]
    for p=1:P
        for k=1:K
            res = res...
                +prior.shape.beta*expected_log_ksi(p)...
                +(prior.shape.beta-1)*expected_log_beta(p,k)...
                -expected_ksi(p)*expected_beta(p,k);
        end
    end
    
    %% E[ln p(tau | mu)]
    for v=1:U
        for u=inedges{v}
            res = res+...
                 prior.shape.tau*expected_log_mu(u)...
                +(prior.shape.tau-1)*expected_log_tau(u,v)...
                -expected_mu(u)*expected_tau(u,v); 
        end
    end
    %% E[ln (ksi)]
    for p=1:P
        res = res+...
            (prior.shape.ksi-1)*expected_log_ksi(p)...
            -prior.rate.ksi*expected_ksi(p);
    end
    %% E[ln (eta)]
    for u=1:U
        res = res+...
            (prior.shape.eta-1)*expected_log_eta(u)...
            -prior.rate.eta*expected_eta(u);
    end
    %% E[ln (mu)]
    for u=1:U
        res = res+...
            (prior.shape.mu-1)*expected_log_mu(u)...
            -prior.rate.mu*expected_mu(u);
    end
    %% E[ln p(E,S|theta,beta,tau)]
    %cupijk * ln theta(u,i,k)
    for u=1:U
        for k=1:K
            res = res...
                +cuk(u,k)*expected_log_theta(u,k);
        end
    end
    %cupk * ln beta(p,k)
    for p=1:P
        for k=1:K
            res = res...
                +cpk(p,k)*expected_log_beta(p,k);
        end
    end
    %cuv
    for u=1:U
        for v=inedges{u}
            res = res+cuv(v,u)*expected_log_tau(v,u);
        end
    end
    
    res = res+sum_sij_gw;
    
   %% sigma(theta(u,i)*beta(p,i)*F(i,j,T))
    sumTheta = (T-t0)'*expected_theta; %(1*U) * (U*K) = 1*K
    sumBeta = sum(expected_beta);
    res = res-sumTheta*sumBeta'; %before formula (5) : sigma(Theta(u,i)*Beta(p,j)*F(i,j,T))
    
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