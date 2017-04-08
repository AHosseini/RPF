%this function assumes that g_w(t1+t2) = g_w(t1)*g_w(t2)
function [cuv,cuk,cpk] = linearTimeComputePhi(...
            U,P,K,inedges,eventsMatrix,events,...
            gamma,w,g,g_log,g_factorized)
    cuv = zeros(U,U);
    cuk = zeros(U,K);
    cpk = zeros(P,K);
    for ui = 1:U
        for pi = 1:P
            [cuv,cuk,cpk] = linearTimeComputePhiUserProduct(...
                U,K,ui,pi,inedges,eventsMatrix,...
                cuv,cuk,cpk,...
                gamma,w,g,g_factorized);
        end
    end
end

function [cuv,cuk,cpk] = linearTimeComputePhiUserProduct(...
            U,K,ui,pi,inedges,eventsMatrix,...
            cuv,cuk,cpk,...
            gamma,w,g,g_factorized)
    %n = length(eventsMatrix{ui,pi}});
    [all_events_length, all_events_users, all_events_times, all_events_f, all_events_label] = ...
        computeAllEventsWithLabels(U,ui,pi,inedges,eventsMatrix,gamma,w,g);
    n = length(eventsMatrix{ui,pi});
    if n == 0
        return;
    end
    G = zeros(1,n);
    for i=1:all_events_length
        G(all_events_label(i)) = G(all_events_label(i))+all_events_f(i);
    end
    for j = 2:n
        G(j) = G(j)+G(j-1)*g_factorized(eventsMatrix{ui,pi}(j)-eventsMatrix{ui,pi}(j-1),w,1);
    end
    
    %
    phi_negative = zeros(1,K);
    for k=1:K
        expected_ln_theta = psi(gamma.theta_shp(ui,k)) - log(gamma.theta_rte(ui,k));
        expected_ln_beta  = psi(gamma.beta_shp(pi,k) ) - log(gamma.beta_rte(pi,k));
        
        phi_negative(k) = exp(expected_ln_theta + expected_ln_beta);
    end
    C = sum(phi_negative);
    
    % 
    P_negative = 0.0;
    for j=1:n
        P_negative = P_negative + (1.0/(C+G(j)));
    end
    for k=1:K
        phi_negative(k) = phi_negative(k)*P_negative;
        cuk(ui,k) = cuk(ui,k)+phi_negative(k);
        cpk(pi,k) = cpk(pi,k)+phi_negative(k);
    end
    
    P_positive = zeros(1,n);
    P_positive(n) = 1.0/(C+G(n));
    
    for j=(n-1):-1:1
        P_positive(j) = 1.0/(C+G(j))+P_positive(j+1)*g_factorized(eventsMatrix{ui,pi}(j+1)-eventsMatrix{ui,pi}(j),w,1);
    end
    phi_positive = zeros(1,all_events_length);
    for j=1:all_events_length
        phi_positive(j) = all_events_f(j)*P_positive(all_events_label(j));
        %{
        label = all_events_label(j);
        for i=label:n
            phi_positive(j) = phi_positive(j)+...
                all_events_f(j)*(1.0/(C+G(i)))*g(eventsMatrix{ui,pi}(i)-eventsMatrix{ui,pi}(label),w);
        end
        %}
        uj = all_events_users(j);        
        cuv(uj,ui) = cuv(uj,ui)+phi_positive(j);
    end
end

function [all_events_length, all_events_users, all_events_times, all_events_f, all_events_label] = ...
    computeAllEventsWithLabels(U,ui,pi,inedges,eventsMatrix,gamma,w,g)

    num_events = 0.0;
    for uj=inedges{ui}
        num_events = num_events+length(eventsMatrix{uj,pi});
    end
    
    all_events_length = 0;
    all_events_users = zeros(1,num_events);
    all_events_times = zeros(1,num_events);
    all_events_f = zeros(1,num_events);
    all_events_label = zeros(1,num_events);
    
     exp_expected_ln_tau = zeros(1,U);
    for uj=inedges{ui}
        exp_expected_ln_tau(uj) = exp(psi(gamma.tau_shp(uj,ui))-log(gamma.tau_rte(uj,ui)));
    end
    
    ptr = ones(U,1);    
    for i = 1:length(eventsMatrix{ui,pi})
        ti = eventsMatrix{ui,pi}(i);
        for uj = inedges{ui}
            while (ptr(uj) <= length(eventsMatrix{uj,pi}))
                tj = eventsMatrix{uj,pi}(ptr(uj));
                if (tj >= ti) %only events before ti can influence on event i
                    break;
                end
                all_events_length = all_events_length+1;
                all_events_users(all_events_length) = uj;
                all_events_times(all_events_length) = tj;
                all_events_f(all_events_length) = exp_expected_ln_tau(uj)*g(ti-tj,w);
                all_events_label(all_events_length) = i;
                ptr(uj) = ptr(uj)+1;
            end
            
        end
    end
    
    all_events_users = all_events_users(1:all_events_length);
    all_events_times = all_events_times(1:all_events_length);
    all_events_f = all_events_f(1:all_events_length);
    all_events_label = all_events_label(1:all_events_length);
       
end

