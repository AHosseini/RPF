function InterEventIntensityIntegrals = NonDynamicInterEventIntensityIntegral(...
                                events, theta, beta, tau, outedges, w, g, params)
U = params.U;
N = length(events);

InterEventIntensityIntegrals = zeros(N-1,1);
socialIntensity = zeros(U,1);

theta_sum = squeeze(sum(theta,1))';
beta_sum = squeeze(sum(beta,1))';
tau_u = zeros(U,1);
for u = 1:U
    tau_u(u) = sum(tau(u,outedges{u}));
end
for n = 1:N-1
    t1 = events{n}.time;
    t2 = events{n+1}.time;
    u = events{n}.user;
    if mod(n,10000)==1
        fprintf('n=%d\n',n-1);
    end
    if n>1
        lastT = events{n-1}.time;
        socialIntensity = socialIntensity * g(t1-lastT,w);
    end
    socialIntensity(u)=socialIntensity(u)+1;
    InterEventIntensityIntegrals(n) = InterEventIntensityIntegrals(n)+...
                theta_sum'*beta_sum*(t2-t1);
    InterEventIntensityIntegrals(n) = InterEventIntensityIntegrals(n)+...
        (1-exp(-1*w*(t2-t1)))*tau_u'*socialIntensity;
end 
end