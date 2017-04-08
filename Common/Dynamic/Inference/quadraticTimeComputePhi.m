%%% quadraticTimeComputePhi
function [cuv, cuik, cpjk,sum_entropy_multinomial,sum_sij_gw] = ...
    quadraticTimeComputePhi(eventsMatrix, events, inedges, gamma, w, g, g_log, params)
    U = params.U;
    P = params.P;
    K = params.K;
    I = params.I;
    J = params.J;
    
    cuv = zeros(U,U);
    cuik = zeros(U,I,K);
    cpjk = zeros(P,J,K);
    
    sum_entropy_multinomial = 0.0;
    sum_sij_gw = 0.0;
    
    psi_gamma.theta_shp = psi(gamma.theta_shp);
    psi_gamma.beta_shp = psi(gamma.beta_shp);
    psi_gamma.tau_shp = psi(gamma.tau_shp);
    log_gamma.theta_rte=log(gamma.theta_rte);
    log_gamma.beta_rte = log(gamma.beta_rte);
    log_gamma.tau_rte = log(gamma.tau_rte);
    
    for n=1:length(events)
        tn = events{n}.time;
        un = events{n}.user;
        pn = events{n}.product;
        [valid_events_length,valid_events_users,valid_events_times] = ...
            computeValidEvents(tn,un,pn,inedges,eventsMatrix);
        
        [valid_features_length , valid_features_i , valid_features_j]  = computeValidFeatures(tn);
        
        [log_phi_negative , log_phi_positive] = computeLogPhi(...
            U,un,tn,pn,inedges,...
            valid_features_length,valid_features_i,valid_features_j,...
            K,psi_gamma,log_gamma,valid_events_users,valid_events_times,g_log,w);
        
        [phi_negative , phi_positive] = computePhi(log_phi_negative,log_phi_positive);
        [sum_entropy_multinomial,sum_sij_gw] = updateEntropyAndSG(...
            sum_entropy_multinomial,sum_sij_gw,...
            phi_negative,phi_positive,...
            w,g_log,tn,valid_events_times);
        
        [cuv,cuik,cpjk] = updateCounts(un,pn,cuv,cuik,cpjk,...
            valid_events_length,valid_events_users,...
            valid_features_length , valid_features_i , valid_features_j,...
            phi_negative,phi_positive,K);
    end
end

%%% compute Phi
function [phi_negative , phi_positive] = computePhi(log_phi_negative,log_phi_positive)
    max_log_phi = max(log_phi_negative);
    if (~isempty(log_phi_positive))
        max_log_phi = max(max_log_phi,max(log_phi_positive));
    end
    
    phi_negative = exp(log_phi_negative-max_log_phi);
    phi_positive = exp(log_phi_positive-max_log_phi);
    
    sum_phi = sum(phi_negative)+sum(phi_positive);
    
    phi_negative = phi_negative/sum_phi;
    phi_positive = phi_positive/sum_phi;
end

%%% compute Log Phi
function [log_phi_negative , log_phi_positive] = computeLogPhi(...
            U,un,tn,pn,inedges,...
            valid_features_length,valid_features_i,valid_features_j,...
            K,psi_gamma, log_gamma, valid_events_users,valid_events_times,g_log,w)
        
    cnt = 0;
    log_phi_negative = zeros(K*valid_features_length,1);
    for m=1:valid_features_length
        im = valid_features_i(m);
        jm = valid_features_j(m);
        expected_ln_theta = psi_gamma.theta_shp(un,im,1:K) - log_gamma.theta_rte(un,im,1:K);
        expected_ln_beta  = psi_gamma.beta_shp(pn,jm,1:K) - log_gamma.beta_rte(pn,jm,1:K);
        log_phi_negative(cnt+1:cnt+K) = expected_ln_theta + expected_ln_beta; 
        cnt = cnt + K;
    end
    expected_ln_tau = zeros(1,U);
    expected_ln_tau(inedges{un}) = psi_gamma.tau_shp(inedges{un},un)-log_gamma.tau_rte(inedges{un},un);
    log_phi_positive = (expected_ln_tau(valid_events_users) + g_log(tn-valid_events_times,w))';
end

%%% computeValidFeatures
%valid features are features that lambda(t,s) !=0.
%so  h_i(tn) , l_j(tn) != 0.
%because it's independet of K. We only save i(s) and j(s).
function [valid_features_length, valid_features_i, valid_features_j] = ...
    computeValidFeatures(tn)
    [day,hour] = dayAndHour(tn);
    valid_features_length = 4;
    valid_features_i = [day+1, day+1, hour+8,hour+8];
    valid_features_j = [day+1, hour+8,day+1, hour+8];    
end


%Valid events are events that can influence on event(tn,un,pn)
%So each event (tm,um,pm) such that um can influence on un,
% & tm < tn & pm = pn is a valid event.
function [valid_events_length,valid_events_users,valid_events_times] = ...
    computeValidEvents(tn,un,pn,inedges,eventsMatrix)
    
num_events = 0.0;
for um=inedges{un}
    num_events = num_events+length(eventsMatrix{um,pn});
end
valid_events_length = 0;
valid_events_users = zeros(1,num_events);
valid_events_times = zeros(1,num_events);
for um=inedges{un}
    indices = find(eventsMatrix{um,pn}<tn);
    num = length(indices);
    valid_events_users(valid_events_length+1:valid_events_length+num) = um;
    valid_events_times(valid_events_length+1:valid_events_length+num) = eventsMatrix{um,pn}(indices);
    valid_events_length = valid_events_length + num;
end
valid_events_users = valid_events_users(1:valid_events_length);
valid_events_times = valid_events_times(1:valid_events_length);

end

%%% updateCounts
function [cuv,cuik,cpjk] = ...
    updateCounts(un,pn,cuv,cuik,cpjk,...
            valid_events_length,valid_events_users,...
            valid_features_length , valid_features_i , valid_features_j,...
            phi_negative,phi_positive,K)
    cnt = 0;
    for m=1:valid_features_length
        im = valid_features_i(m);
        jm = valid_features_j(m);
        cuik(un,im,1:K) = cuik(un,im,1:K)+reshape(phi_negative(cnt+1:cnt+K),[1,1,K]);
        cpjk(pn,jm,1:K) = cpjk(pn,jm,1:K)+reshape(phi_negative(cnt+1:cnt+K),[1,1,K]);        
        cnt = cnt+K;
    end
    if (valid_events_length > 0)
        for m=1:valid_events_length
           um = valid_events_users(m);
           cuv(um,un) = cuv(um,un)+phi_positive(m);
        end
    end    
end

function [sum_entropy_multinomial,sum_sij_gw] = updateEntropyAndSG(...
            sum_entropy_multinomial,sum_sij_gw,...
            phi_negative,phi_positive,...
            w, g_log,tn,valid_events_times)
        neg_indices = phi_negative >1e-40;
        sum_entropy_multinomial = sum_entropy_multinomial -  sum(phi_negative(neg_indices).*log(phi_negative(neg_indices)));
        pos_indices = phi_positive >1e-40;
        if ~isempty(phi_positive)
            sum_entropy_multinomial = sum_entropy_multinomial -  sum(phi_positive(pos_indices).*log(phi_positive(pos_indices)));
            sum_sij_gw = sum_sij_gw+g_log(tn-valid_events_times,w)*phi_positive;
        end
end
