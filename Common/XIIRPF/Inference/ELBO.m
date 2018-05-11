function res = ELBO(T,itemsCluster, itemDistances,...
            inedges,SUMG_User,SUMG_User_Product,...
            sum_sij_gw,sum_sij_d,sum_entropy_multinomial,...
            prior,cuk,cpk,cuv,cbb,cu_bar,t0,gamma,kernel,params)
        U=params.U;
        P=params.P;
        B=params.B;
        K=params.K;
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
    expected_phi = gamma.phi_shp ./ gamma.phi_rte;
    expected_log_phi =(psi(gamma.phi_shp) - log(gamma.phi_rte));
    expected_pi = gamma.pi_shp ./ gamma.pi_rte;
    expected_log_pi =(psi(gamma.pi_shp) - log(gamma.pi_rte));
    
    expected_ksi = gamma.ksi_shp ./ gamma.ksi_rte;
    expected_log_ksi = (psi(gamma.ksi_shp) - log(gamma.ksi_rte));
    expected_eta  = gamma.eta_shp  ./ gamma.eta_rte;
    expected_log_eta  = (psi(gamma.eta_shp)  - log(gamma.eta_rte));
    expected_mu  = gamma.mu_shp  ./ gamma.mu_rte;
    expected_log_mu  = (psi(gamma.mu_shp)  - log(gamma.mu_rte));
    
    expected_psi  = gamma.psi_shp  ./ gamma.psi_rte;
    expected_log_psi  = (psi(gamma.psi_shp)  - log(gamma.psi_rte));
    expected_rho  = gamma.rho_shp  ./ gamma.rho_rte;
    expected_log_rho  = (psi(gamma.rho_shp)  - log(gamma.rho_rte));
    
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
    %% E[ln p(phi | psi)]
    for u=1:U
        res = res+...
             prior.shape.phi*expected_log_psi(u)...
            +(prior.shape.phi-1)*expected_log_phi(u)...
            -expected_psi(u)*expected_phi(u); 
    end
    %% E[ln p(pi | rho)]
%     res_old = res;
    for b=1:B
        for c = 1:B
            res = res+...
                 prior.shape.pi*expected_log_rho(b)...
                +(prior.shape.pi-1)*expected_log_pi(b,c)...
                -expected_rho(b)*expected_pi(b,c); 
        end
    end
%     fprintf('E[ln p(pi | rho)] is %f\n',res-res_old);
    %% E[ln (ksi)]
    for p=1:P
        res = res+...
            (prior.shape.ksi-1)*expected_log_ksi(p)...
            -prior.rate.ksi*expected_ksi(p);
    end
    %% E[ln (rho)]
%     res_old = res;
    for b=1:B
        res = res+...
            (prior.shape.rho-1)*expected_log_rho(b)...
            -prior.rate.rho*expected_rho(b);
    end
%     fprintf('E[ln (rho)] is %f\n',res-res_old);
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
    %% E[ln (psi)]
    for u=1:U
        res = res+...
            (prior.shape.psi-1)*expected_log_psi(u)...
            -prior.rate.psi*expected_psi(u);
    end
    %% E[ln p(E,S|theta,beta,tau,phi,pi)]
    %cuk * ln theta(u,k)
    res = res+sum(sum(cuk.*expected_log_theta));
    %cpk * ln theta(p,k)
    res = res + sum(sum(cpk .* expected_log_beta));
    %cuv * ln tau(u,v)
    res = res + sum(sum(cuv.*expected_log_tau));
    %c_bar(u)* ln phi_u
    res = res + sum(cu_bar.*expected_log_phi);
    %cbb'* ln pi_bb'
    res = res + sum(sum(cbb.*expected_log_pi));
    res = res+sum_sij_gw;
    res = res+sum_sij_d;
%     fprintf('E[ln p(E,S|theta,beta,tau)]=%f\n',res-res_old);
    
   %% sigma(theta(u,i)*beta(p,i)*(T-t0))
    sumTheta = (T-t0)'*expected_theta; %(1*U) * (U*K) = 1*K
    sumBeta = sum(expected_beta);
    res = res-sumTheta*sumBeta'; %before formula (5) : sigma(Theta(u,i)*Beta(p,j)*F(i,j,T))
    
    %% sigma (tau(v,u) * G(T-ti))
    for v=1:U
        for u=inedges{v}
            res = res - expected_tau(u,v)*SUMG_User(u,v);
        end
    end
    %% sigma E[phi_u]E[Bpe](Sigma E[Bp])Gw(T-ti)
    for p=1:P
        alpha_p = expected_pi(itemsCluster(p),itemsCluster(1:P))*kernel.d(itemDistances(p,:))'...
            -expected_pi(itemsCluster(p),itemsCluster(p))*kernel.d(itemDistances(p,p))';
        res = res-alpha_p*SUMG_User_Product(:,p)'*expected_phi;
    end
    %% ENTROPY
    res = res+sum(sum(sum(gamma_entropy(gamma.beta_shp ,gamma.beta_rte))));
    res = res+sum(sum(sum(gamma_entropy(gamma.theta_shp,gamma.theta_rte))));
    for v=1:U
        for u=inedges{v}
            res = res+gamma_entropy(gamma.tau_shp(u,v),gamma.tau_rte(u,v));
        end
    end
    res = res+sum(gamma_entropy(gamma.phi_shp , gamma.phi_rte));
    res = res+sum(sum(gamma_entropy(gamma.pi_shp  , gamma.pi_rte)));
    res = res+sum(gamma_entropy(gamma.ksi_shp , gamma.ksi_rte));
    res = res+sum(gamma_entropy(gamma.eta_shp , gamma.eta_rte));
    res = res+sum(gamma_entropy(gamma.mu_shp  , gamma.mu_rte));
    res = res+sum(gamma_entropy(gamma.psi_shp  , gamma.psi_rte));
    res = res+sum(gamma_entropy(gamma.rho_shp  , gamma.rho_rte));
%     fprintf('sum_entropy_gamma=%f\n',sum(gamma_entropy(gamma.rho_shp  , gamma.rho_rte)));
    res = res+sum_entropy_multinomial;
end

function entropy = gamma_entropy(shp , rte)
    entropy = shp - log(rte) + gammaln(shp) + (1-shp).*psi(shp);
end