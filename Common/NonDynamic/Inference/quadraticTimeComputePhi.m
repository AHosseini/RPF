%%% quadraticTimeComputePhi
function [cuv, cuk, cpk,sum_entropy_multinomial,sum_sij_gw] = ...
    quadraticTimeComputePhi(eventsMatrix, events, inedges, gamma, w, g_log, params)
    U = params.U;
    P = params.P;
    K = params.K;
    
    cuv = zeros(U,U);
    cuk = zeros(U,K);
    cpk = zeros(P,K);
    
    sum_entropy_multinomial = 0.0;
    sum_sij_gw = 0.0;
    
    psi_gamma.theta_shp = psi(gamma.theta_shp);
    psi_gamma.beta_shp = psi(gamma.beta_shp);
    psi_gamma.tau_shp = psi(gamma.tau_shp);
    log_gamma.theta_rte=log(gamma.theta_rte);
    log_gamma.beta_rte = log(gamma.beta_rte);
    log_gamma.tau_rte = log(gamma.tau_rte);
%     valid_events = Cell(U,P);
    
    for n=1:length(events)
        tn = events{n}.time;
        un = events{n}.user;
        pn = events{n}.product;
        
        [valid_events_users,valid_events_times] = ...
            computeValidEvents(tn,un,pn,inedges,eventsMatrix);
        
        
        [log_phi_negative , log_phi_positive] = computeLogPhi(...
            U,un,tn,pn,inedges,...
            psi_gamma,log_gamma,valid_events_users,valid_events_times,g_log,w);
        
        [phi_negative , phi_positive] = computePhi(log_phi_negative,log_phi_positive);
%         [sum_entropy_multinomial_i,sum_sij_gw_i]=updateEntropyAndSG(...
%             phi_negative,phi_positive,...
%             w,g_log,tn,valid_events_times);
%         sum_entropy_multinomial = sum_entropy_multinomial+sum_entropy_multinomial_i
        [sum_entropy_multinomial,sum_sij_gw] = updateEntropyAndSG(...
            sum_entropy_multinomial,sum_sij_gw,...
            phi_negative,phi_positive,...
            w,g_log,tn,valid_events_times);
        
        [cuv,cuk,cpk] = updateCounts(un,pn,cuv,cuk,cpk,...
            valid_events_users,...
            phi_negative,phi_positive);
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
            psi_gamma, log_gamma, valid_events_users,valid_events_times,g_log,w)
        
    log_phi_negative = psi_gamma.theta_shp(un,:) - log_gamma.theta_rte(un,:)+...
                        psi_gamma.beta_shp(pn,:) - log_gamma.beta_rte(pn,:);
    expected_ln_tau = zeros(1,U);
    expected_ln_tau(inedges{un}) = psi_gamma.tau_shp(inedges{un},un)-...
        log_gamma.tau_rte(inedges{un},un);
    log_phi_positive = (expected_ln_tau(valid_events_users) + g_log(tn-valid_events_times,w));
end


%Valid events are events that can influence on event(tn,un,pn)
%So each event (tm,um,pm) such that um can influence on un,
% & tm < tn & pm = pn is a valid event.
function [valid_events_users,valid_events_times] = ...
    computeValidEvents(tn,un,pn,inedges,eventsMatrix)
    
num_events = 0.0;
for um=inedges{un}
    num_events = num_events+length(eventsMatrix{um,pn});
end
valid_events_length = 0;
valid_events_users = zeros(1,num_events);
valid_events_times = zeros(1,num_events);
for um=inedges{un}
    for tm=eventsMatrix{um,pn}
        if (tm >= tn)
            break;
        end
        valid_events_length = valid_events_length+1;
        valid_events_users(valid_events_length) = um;
        valid_events_times(valid_events_length) = tm;
    end
end

valid_events_users = valid_events_users(1:valid_events_length);
valid_events_times = valid_events_times(1:valid_events_length);

end

%%% updateCounts
function [cuv,cuk,cpk] = ...
    updateCounts(un,pn,cuv,cuk,cpk,...
            valid_events_users,...
            phi_negative,phi_positive)
    cuk(un,:) = cuk(un,:) + phi_negative;
    cpk(pn,:) = cpk(pn,:) + phi_negative;
    cuv(valid_events_users,un) = cuv(valid_events_users,un)+phi_positive';
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
            sum_sij_gw = sum_sij_gw+g_log(tn-valid_events_times,w)*phi_positive';
        end
end
