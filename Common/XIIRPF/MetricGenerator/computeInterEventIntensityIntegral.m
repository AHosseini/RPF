function InterEventIntensityIntegrals = computeInterEventIntensityIntegral(...
                                events,itemsCluster,itemsSimilarity,...
                                theta, beta, tau, phi, pi, outedges, kernel, params)
U = params.U;
P = params.P;
N = length(events);

InterEventIntensityIntegrals = zeros(N-1,1);
socialIntensity = 0;
contextualIntensity = 0;

theta_sum = sum(theta);
beta_sum = sum(beta);
tau_u = zeros(U,1);
for u = 1:U
    tau_u(u) = sum(tau(u,outedges{u}));
end

alpha = zeros(P,1);
for p = 1:P
    alpha(p) = pi(itemsCluster(p),itemsCluster(1:P))*kernel.d(itemsSimilarity(p,:))'...
            -pi(itemsCluster(p),itemsCluster(p))*kernel.d(itemsSimilarity(p,p))';    
end

for n = 1:N-1
    t1 = events{n}.time;
    t2 = events{n+1}.time;
    u = events{n}.user;
    p = events{n}.product;
    if mod(n,1000)==1
        fprintf('Computing interEventIntensity for event %d completed.\n',n-1);
    end
    if n>1
        lastT = events{n-1}.time;
        socialIntensity = socialIntensity * kernel.g(t1-lastT,kernel.w1);
        contextualIntensity = contextualIntensity * kernel.g(t1-lastT,kernel.w2);
    end
    socialIntensity=socialIntensity+tau_u(u);
    contextualIntensity = contextualIntensity+phi(u)*alpha(p);
    InterEventIntensityIntegrals(n) = theta_sum*beta_sum'*(t2-t1)+ ...
        kernel.G(t2-t1,kernel.w1)*socialIntensity+kernel.G(t2-t1,kernel.w2)*contextualIntensity;
end 
end