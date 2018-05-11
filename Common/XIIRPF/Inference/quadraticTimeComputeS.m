%%% quadraticTimeComputePhi
function [cuv, cuk, cpk, cbb, cu_bar, sum_entropy_multinomial, sum_sij_gw,sum_sij_d] = ...
    quadraticTimeComputeS(events,userEvents,itemsCluster,itemsDistances,...
    gamma, kernel, params)
   U = params.U;
    P = params.P;
    K = params.K;
    B = params.B;
    
    cuv = zeros(U,U);
    cuk = zeros(U,K);
    cpk = zeros(P,K);
    cbb = zeros(B,B);
    cu_bar = zeros(U,1);
    
    sum_entropy_multinomial = 0.0;
    sum_sij_gw = 0.0;
    sum_sij_d = 0.0;
    
    psi_gamma.theta_shp = psi(gamma.theta_shp);
    psi_gamma.beta_shp = psi(gamma.beta_shp);
    psi_gamma.tau_shp = psi(gamma.tau_shp);
    psi_gamma.phi_shp = psi(gamma.phi_shp);
    psi_gamma.pi_shp = psi(gamma.pi_shp);
    
    log_gamma.theta_rte=log(gamma.theta_rte);
    log_gamma.beta_rte = log(gamma.beta_rte);
    log_gamma.tau_rte = log(gamma.tau_rte);
    log_gamma.phi_rte = log(gamma.phi_rte);
    log_gamma.pi_rte = log(gamma.pi_rte);
    
    expected_ln_theta = psi_gamma.theta_shp-log_gamma.theta_rte;
    expected_ln_beta = psi_gamma.beta_shp-log_gamma.beta_rte;
    expected_ln_tau = psi_gamma.tau_shp-log_gamma.tau_rte;
    expected_ln_phi = psi_gamma.phi_shp-log_gamma.phi_rte;
    expected_ln_pi = psi_gamma.pi_shp-log_gamma.pi_rte;
    
    N = length(events);
    
for u= 1:U
    [valid_utriggers_times,valid_ptriggers_times,valid_ptriggers_products] = findPossibleTriggerers(P,userEvents{u});
    for n = 1:length(userEvents{u}.time)
        tn = userEvents{u}.time(n);
        pn = userEvents{u}.product(n);
        utriggers_times = valid_utriggers_times{n};
        ptriggers_times=valid_ptriggers_times{n};
        ptriggers_products=valid_ptriggers_products{n};
        [log_phi_negative ,log_phi_positive_users,log_phi_positive_products]=...
            computeLogPhi(u,tn,pn,...
            expected_ln_theta,expected_ln_beta,expected_ln_tau,expected_ln_phi,...
            expected_ln_pi,itemsCluster,itemsDistances,utriggers_times,...
            ptriggers_products,ptriggers_times,kernel);

        [phi_negative, phi_positive_users, phi_positive_products] = ...
            computePhi(log_phi_negative,log_phi_positive_users,log_phi_positive_products);
        [cuv,cuk,cpk,cbb,cu_bar] = ...
         updateCounts(u,pn,itemsCluster,cuv,cuk,cpk,cbb,cu_bar,...
                    ptriggers_products,...
                     phi_negative,phi_positive_users,phi_positive_products);
    %      fprintf('diff(cu_bar)=%.2f    diff(cpqk)=%.2f\n',sum(cu_bar-oldCu_bar),sum(sum(sum(cpqk-oldCpqk))));
        [sum_entropy_multinomial,sum_sij_gw,sum_sij_d] = updateEntropyAndSG(itemsDistances,tn,pn,...
            sum_entropy_multinomial,sum_sij_gw,sum_sij_d,...
            phi_negative,phi_positive_users,phi_positive_products,...
            kernel,utriggers_times,ptriggers_times,ptriggers_products);
    end
end
end
%% 
function [valid_utriggers_times,valid_ptriggers_times,valid_ptriggers_products] = findPossibleTriggerers(P,userEvents)
N = length(userEvents.time);
valid_utriggers_times = cell(N,1);
valid_ptriggers_times = cell(N,1);
valid_ptriggers_products = cell(N,1);

product_indices = cell(P,1);
for n = 2:N
    pn = userEvents.product(n);
    indices = ones(1,n-1);
    indices(product_indices{pn}) = 0;
    valid_utriggers_times{n} = userEvents.time(product_indices{pn});
    valid_ptriggers_times{n} = userEvents.time(indices==1);
    valid_ptriggers_products{n} = userEvents.product(indices==1);
    product_indices{pn}(end+1) = n;
end
end

%% compute Phi
function [phi_negative , phi_positive_users, phi_positive_products] = ...
    computePhi(log_phi_negative, log_phi_positive_users, log_phi_positive_products)
    max_log_phi = max(log_phi_negative);
    if (~isempty(log_phi_positive_users))
        max_log_phi = max(max_log_phi,max(log_phi_positive_users));
    end
    if (~isempty(log_phi_positive_products))
        max_log_phi = max(max_log_phi,max(max(log_phi_positive_products)));
    end
    
    phi_negative = exp(log_phi_negative-max_log_phi);
    phi_positive_users = exp(log_phi_positive_users-max_log_phi);
    phi_positive_products = exp(log_phi_positive_products-max_log_phi);
    
    sum_phi = sum(phi_negative)+sum(phi_positive_users)+sum(sum(phi_positive_products));
    
    phi_negative = phi_negative/sum_phi;
    phi_positive_users = phi_positive_users/sum_phi;
    phi_positive_products = phi_positive_products/sum_phi;
end

%%% compute Log Phi
function [log_phi_negative , log_phi_positive_users,log_phi_positive_products] = ...
        computeLogPhi(un,tn,pn,...
        expected_ln_theta,expected_ln_beta,expected_ln_tau,expected_ln_phi,expected_ln_pi, ...
        itemsCluster, itemsDistances,...
        valid_utriggers_times,valid_ptriggers_products,valid_ptriggers_times,...
        kernel)
    log_phi_negative = expected_ln_theta(un,:)+expected_ln_beta(pn,:);
    
    log_phi_positive_users = [];
    if ~isempty(valid_utriggers_times)
        log_phi_positive_users = expected_ln_tau(un,un) + kernel.g_log(tn-valid_utriggers_times,kernel.w1);
    end
    
    log_phi_positive_products = [];
    
    if ~isempty(valid_ptriggers_times)
        log_phi_positive_products = expected_ln_phi(un)+...
            expected_ln_pi(itemsCluster(pn),itemsCluster(valid_ptriggers_products))+...
            kernel.d_log(itemsDistances(pn,valid_ptriggers_products))+kernel.g_log(tn-valid_ptriggers_times,kernel.w2);
    end
end

function [cuv,cuk,cpk,cbb,cu_bar] = ...
        updateCounts(un,pn,itemsCluster,cuv,cuk,cpk,cbb,cu_bar,...
                valid_ptriggers_products,...
                 phi_negative,phi_positive_users,phi_positive_products)
    cuk(un,:) = cuk(un,:) + phi_negative;
    cpk(pn,:) = cpk(pn,:) + phi_negative;
    %cuv(valid_events_users,un) = cuv(valid_events_users,un)+phi_positive';
    cuv(un,un) = cuv(un,un) + sum(phi_positive_users);
    for m=1:length(valid_ptriggers_products)
        pm = valid_ptriggers_products(m);
        cbb(itemsCluster(pn),itemsCluster(pm)) = cbb(itemsCluster(pn),itemsCluster(pm)) + phi_positive_products(m);
    end
    cu_bar(un) = cu_bar(un) + sum(sum(phi_positive_products));
%     if un==1089
%         fprintf('pn = %d, sum(phi_negative)=%f, cu_bar(1089)=%f\n',pn, sum(phi_negative), sum(sum(phi_positive_products)));
%     end
end


function [sum_entropy_multinomial,sum_sij_gw,sum_sij_d] = updateEntropyAndSG(itemsDistances,tn,pn,...
            sum_entropy_multinomial,sum_sij_gw,sum_sij_d,...
            phi_negative,phi_positive_users,phi_positive_products,...
            kernel,valid_utriggers_times,valid_ptriggers_times,valid_ptriggers_products)
        neg_indices = phi_negative >1e-40;
        if sum(neg_indices)>0
            sum_entropy_multinomial = sum_entropy_multinomial -  sum(phi_negative(neg_indices).*log(phi_negative(neg_indices)));
        end
        pos_indices = phi_positive_users >1e-40;
        if sum(pos_indices)>0
            sum_entropy_multinomial = sum_entropy_multinomial -  sum(phi_positive_users(pos_indices).*log(phi_positive_users(pos_indices)));
            sum_sij_gw = sum_sij_gw+kernel.g_log(tn-valid_utriggers_times,kernel.w1)*phi_positive_users';
        end
        pos_indices_products = phi_positive_products > 1e-40;
        if sum(pos_indices_products)>0
            temp = phi_positive_products(pos_indices_products);
            sum_entropy_multinomial = sum_entropy_multinomial - sum(temp.*log(temp));
            sum_sij_gw = sum_sij_gw+kernel.g_log(tn-valid_ptriggers_times,kernel.w2)*phi_positive_products';
            sum_sij_d = sum_sij_d + kernel.d_log(itemsDistances(pn,valid_ptriggers_products))*phi_positive_products';
        end
end