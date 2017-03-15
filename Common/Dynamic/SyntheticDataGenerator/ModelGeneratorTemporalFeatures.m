function model = ModelGeneratorTemporalFeatures(U,P,K,I,sp)
%inputs :
%   U : number of users
%   P : number of products
%   K : dimension of each user or product preferences
%   I : number of feature vectors
%   sp: sparsity of graph (default = 0.1)
%outputs:
%   a struct model containing following fields
%       U       : number of users
%       P       : number of products
%       theta   : array of cells containing each user's vector of properties
%       beta    : array of cells containing product's vector of properties
%       tau     : array of cells containing outgoing edges from each user
%       (if a user u follows user v , then v can influence on u , so we
%       save tau(u,v))
%       tauInvert:array of cells containing incoming edges from each user
%       a       : whole graph

%input parameters checking

if (nargin < 4)
    disp('usage: modelGenerator(U, P, K, I, sp)');
    return;
end
if (nargin < 5)
    sp = 0.1;
end

%model initialization 
model = struct;
% rng(1);
model.rng = rng;
model.U = U;
model.P = P;
model.K = K;
model.I = I;
model.eta = cell(U,1);
model.theta = cell(U, I );
model.ksi = cell(P,1);
model.beta = cell(P, I );
model.mu = cell(U,1);
model.tau = cell(U,1);
model.tauInvert = cell(U,1);

% Prior parameters
model.theta_prior_shape = 1;
model.beta_prior_shape = 1;
model.tau_prior_shape = 1;
model.mu_prior_rate = 0.1;
model.mu_prior_shape = 1;
model.eta_prior_rate = 0.1;
model.eta_prior_shape = 1;
model.ksi_prior_rate = 0.1;
model.ksi_prior_shape = 1;

%tmp
tmp = [.9 .05 .05; .05 .9 .05 ; .05 .05 .9];
%theta
for u=1:U
    model.eta{u} = random('Gamma',model.eta_prior_shape,1/model.eta_prior_rate);
    for i=1:I
        model.theta{u,i} = random('Gamma',model.theta_prior_shape,1/model.eta{u},K,1);
    end
    model.mu{u} = random('Gamma',model.mu_prior_shape,1/model.mu_prior_rate);
end
%beta
for p=1:P
    model.ksi{p} = random('Gamma',model.ksi_prior_shape,1/model.ksi_prior_rate);
    for i=1:I
        model.beta{p,i} = random('Gamma',model.beta_prior_shape,1/model.ksi{p},K,1);
    end
end

%tau
model.a = sparse(U,U);

M = floor(sp*U^2);
edges = datasample(0:U^2-1,M,'Replace',false);

for i=1:M
    head = mod(edges(i),U)+1;
    tail = fix(edges(i)/U)+1;
    model.a(head,tail) =random('Gamma',model.tau_prior_shape,1/model.mu{head});
end

for u=1:U
    model.tau{u} = model.a(u,:);
    model.tauInvert{u} = model.a(:,u)';
end
end