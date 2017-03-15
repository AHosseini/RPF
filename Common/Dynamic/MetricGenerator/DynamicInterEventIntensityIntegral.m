function InterEventIntensityIntegrals = DynamicInterEventIntensityIntegral(...
                                events, theta, beta, tau, outedges, w, g, params)
U = params.U;
P = params.P;
J = params.J;
I = params.I;
N = length(events);

InterEventIntensityIntegrals = zeros(N-1,1);
socialIntensity = zeros(U,1);

theta_i = squeeze(sum(theta,1))';
beta_j = squeeze(sum(beta,1))';
tau_u = zeros(U,1);
F = zeros(I,J);
lastF = zeros(I,J);
for i=1:I
    for j=1:J
        F(i,j) = integralOfProduct(i,j,events{1}.time);
    end
end
for u = 1:U
    tau_u(u) = sum(tau(u,outedges{u}));
end
for n = 1:N-1
    t1 = events{n}.time;
    t2 = events{n+1}.time;
    u = events{n}.user;
    if mod(n,10000)==1
        fprintf('n=%d\n',n);
    end
    if n>1
        lastT = events{n-1}.time;
        socialIntensity = socialIntensity * g(t1-lastT,w);
    end
    socialIntensity(u)=socialIntensity(u)+1;
    for i=1:I
        for j=1:J
            lastF(i,j) = F(i,j);
            F(i,j) = integralOfProduct(i,j,t2);
            InterEventIntensityIntegrals(n) = InterEventIntensityIntegrals(n)+...
                theta_i(:,i)'*beta_j(:,j)*(F(i,j)-lastF(i,j));
        end
    end
    InterEventIntensityIntegrals(n) = InterEventIntensityIntegrals(n)+...
        (1-exp(-1*w*(t2-t1)))*tau_u'*socialIntensity;
end 
end