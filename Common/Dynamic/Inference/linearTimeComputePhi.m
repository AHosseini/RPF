%% linearTimeComputePhi
%this function assumes that g_w(t1+t2) = g_w(t1)*g_w(t2)
function [cuv,cuik,cpjk] = linearTimeComputePhi(...
            U,P,K,I,J,inedges,eventsMatrix,events,...
            gamma,w,g,g_factorized)
    cuv = zeros(U,U);
    cuik = zeros(U,I,K);
    cpjk = zeros(P,J,K);
    
    expected_ln_theta_matrix = psi(gamma.theta_shp)-log(gamma.theta_rte);
    expected_ln_beta_matrix = psi(gamma.beta_shp)-log(gamma.beta_rte);
    
    for ui = 1:U
        for pi = 1:P
            [cuv,cuik,cpjk] = linearTimeComputePhiUserProduct(...
                U,K,I,J,ui,pi,inedges,eventsMatrix,...
                cuv,cuik,cpjk,...
                gamma,w,g,g_factorized,expected_ln_theta_matrix,expected_ln_beta_matrix);
        end
    end
end

function [cuv,cuik,cpjk] = linearTimeComputePhiUserProduct(...
                U,K,I,J,u,p,inedges,eventsMatrix,...
                cuv,cuik,cpjk,...
                gamma,w,g,g_factorized,expected_ln_theta_matrix,expected_ln_beta_matrix);
    [all_events_length, all_events_users, all_events_times, all_events_f, all_events_label] = ...
        computeAllEventsWithLabels(U,u,p,inedges,eventsMatrix,gamma,w,g);
    n = length(eventsMatrix{u,p});
    if n == 0
        return;
    end
    %% G
    G = zeros(1,n);
    for i=1:all_events_length
        G(all_events_label(i)) = G(all_events_label(i))+all_events_f(i);
    end
    for j = 2:n
        G(j) = G(j)+G(j-1)*g_factorized(eventsMatrix{u,p}(j)-eventsMatrix{u,p}(j-1),w,1);
    end
    
    %% C
    C = zeros(n,1);
    
    for i=1:n
        ti = eventsMatrix{u,p}(i);
        [valid_features_length , valid_features_i , valid_features_j]  = computeValidFeatures(ti);
        phi_negative_weight = zeros(K*valid_features_length,1);
        phi_negative = zeros(K*valid_features_length,1);        
        cnt = 0;
        for m=1:valid_features_length
            im = valid_features_i(m);
            jm = valid_features_j(m);
            for k=1:K
                cnt = cnt+1;
                expected_ln_theta = expected_ln_theta_matrix(u,im,k);
                expected_ln_beta = expected_ln_beta_matrix(p,jm,k);
                phi_negative_weight(cnt) = exp(expected_ln_theta + expected_ln_beta); 
                C(i) = C(i)+phi_negative_weight(cnt);
            end
        end
        %normalize phi_negative_weights
        cnt = 0;
        for m=1:valid_features_length
            im = valid_features_i(m);
            jm = valid_features_j(m);            
            for k=1:K
                cnt = cnt+1;
                phi_negative(cnt) = phi_negative_weight(cnt)/(C(i)+G(i));
                cuik(u,im,k) = cuik(u,im,k)+phi_negative(cnt);
                cpjk(p,jm,k)  = cpjk(p,jm,k)+phi_negative(cnt);
            end
        end
    end
    %% phi positive & P_positive
    P_positive = zeros(1,n);
    P_positive(n) = 1.0/(C(n)+G(n));
    for j=(n-1):-1:1
        P_positive(j) = 1.0/(C(j)+G(j)) +...
            P_positive(j+1)*g_factorized(eventsMatrix{u,p}(j+1)-eventsMatrix{u,p}(j),w,1);
    end
    
    phi_positive = zeros(1,all_events_length);
    for j=1:all_events_length
        phi_positive(j) = all_events_f(j)*P_positive(all_events_label(j));
        uj = all_events_users(j);        
        cuv(uj,u) = cuv(uj,u)+phi_positive(j);
    end        
end

%% computeValidFeatures
% Valid features are features that lambda(t,s) !=0.
% so  h_i(tn) , l_j(tn) != 0.
% because it's independet of K. We only save i(s) and j(s).
function [valid_features_length, valid_features_i, valid_features_j] = ...
    computeValidFeatures(tn)
    [day,hour] = dayAndHour(tn);
    valid_features_length = 4;
    valid_features_i = [day+1, day+1, hour+8,hour+8];
    valid_features_j = [day+1, hour+8,day+1, hour+8];    
end

%% computeAllEventsWithLabels
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
                if (tj >= ti) %only events before ti can 
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
    
    %there may be events after ui,pi , so all_events_length may be smaller
    %than num_events
    all_events_users = all_events_users(1:all_events_length);
    all_events_times = all_events_times(1:all_events_length);
    all_events_f = all_events_f(1:all_events_length);
    all_events_label = all_events_label(1:all_events_length);
       
end

