function [events,eventsMatrix] = PurchaseSimulatorTemporalFeatures(model, iter, w, g)

if nargin < 2
    iter = 10*1000;
end
if nargin < 3
    w = 1;
end
if nargin < 4
   g = @(x,w) w*exp(-w*x);
   g_factorized = @(x,w,p) p*exp(-w*x);
end

%initialization
t = 0;
n = 0;
U = model.U;
P = model.P;
K = model.K;

events = cell(iter,1);
eventsMatrix = cell(U,P);

suprimumThetaBeta = computeSuprimumThetaBeta(model);
socialIntensity = zeros(U,P);

baseIntensityOverDayAndHour = zeros(U,P,7,24);

thetau = zeros(U,K);
betap = zeros(P,K);

for d=1:7
    for h=1:24
        for u=1:U
            thetau(u,:) = model.theta{u,d}+model.theta{u,h+7};
        end
        for p=1:P
            betap(p,:) = model.beta{p,d}+model.beta{p,h+7};
        end
        baseIntensityOverDayAndHour(:,:,d,h) = thetau * betap';
    end
end

fprintf('now starting generating points , t = %f',t);
%cnt = 0;
%% loop
while n < iter
    %cnt = cnt+1
%    I = SuprimumPurchaseIntensityTemporalFeatures(model,eventsMatrix,suprimumThetaBeta,t,w,g);
    newI = suprimumThetaBeta+socialIntensity;
 %   fprintf('error newI with I = %f\n',sum(sum((newI-I).^2)));
    if (mod(n,10000) == 0)
        disp(n);
    end
    sumI = sum(sum(newI));
    
    deltaT = exprnd(1/sumI);
    t = t+deltaT;
    socialIntensity = socialIntensity*g_factorized(deltaT,w,1);
    
    %[baseIntensity] = baseIntensityModel(model,t);
    [d,h] = dayAndHour(t);
    %fprintf('diff baseintensity for day %d , hour %d , = %f\n',d,h,sum(sum((baseIntensity-baseIntensityOverDayAndHour(:,:,d+1,h+1)).^2 )));
    baseIntensity = baseIntensityOverDayAndHour(:,:,d+1,h+1);
    
    %Is = PurchaseIntensityTemporalFeatures(model,eventsMatrix,t,w,g,baseIntensity);
    newIs = baseIntensity+socialIntensity;
%    fprintf('error newIs with Is = %f\n',sum(sum((newIs-Is).^2)));
    
    sumS = sum(sum(newIs));    
    c = rand();
    if (c*sumI < sumS)
        n = n+1;
        event = struct;
        event.time = t;
        [u,p] = sample(newIs);
        event.user = u;
        event.product = p;
        events{n} = event;
        eventsMatrix{u,p}(end+1) = t;
        [~,V,TAU] = find(model.tau{u});
        for i=1:length(V)
            v = V(i);
            tau = TAU(i);
            socialIntensity(v,p) = socialIntensity(v,p)+tau*g(0,w);
        end
        %disp(['t = ',num2str(t*1000),' u = ',num2str(u),' p =' ,num2str(p)]);
    end
end
end

%% computeSuprimumThetaBeta
function [suprimumThetaBeta] = computeSuprimumThetaBeta(model)
    suprimumThetaBeta = zeros(model.U,model.P);
    for u=1:model.U
        if (mod(u,50) == 1)
            disp(u);
        end
        for p=1:model.P
            maxThetaBeta = 0.0;
            for i=1:7
                for j=8:31
                    theta = model.theta{u,i}+model.theta{u,j};
                    beta  = model.beta{p,i} +model.beta{p,j};
                    maxThetaBeta = max(maxThetaBeta,theta'*beta);
                end
            end
            suprimumThetaBeta(u,p) = maxThetaBeta;
        end
    end
end

function [baseIntensity] = baseIntensityModel(model,t)

theta = cell(model.U,1);
for u=1:model.U
    theta{u} = zeros(model.K,1);
    cnt = 0;
    for i=1:model.I
        if (h(t,i))
            cnt = cnt+1;
            theta{u} = theta{u}+model.theta{u,i};
        end
    end
end

beta = cell(model.P,1);
for p=1:model.P
    beta{p} = zeros(model.K,1);
    for i=1:model.I
        if (l(t,i))
            beta{p} = beta{p}+model.beta{p,i};
        end
    end
end

baseIntensity = zeros(model.U,model.P);
for u=1:model.U
    for p=1:model.P
        baseIntensity(u,p) = theta{u}'*beta{p};
    end
end
end

function isInRange = h(t,i)
    isInRange = 0;
    [day,hour] = dayAndHour(t);
    if (1 <= i && i <= 7)
        i = i-1; %do this to simplify calculation in mod 7
        %day = mod(floor(t/24),7);
        isInRange = (day == i);
    else
        i = i-8; %do this to simplify calculation in mod 24
        %hour = mod(floor(t),24);
        isInRange = (hour == i);
    end
end

function isInRange = l(t,i)
isInRange = 0;
    if (1 <= i && i <= 7)
        i = i-1; %do this to simplify calculation in mod 7
        day = mod(floor(t/24),7);
        isInRange = (day == i);
    else
        i = i-8; %do this to simplify calculation in mod 24
        hour = mod(floor(t),24);
        isInRange = (hour == i);
    end
end
