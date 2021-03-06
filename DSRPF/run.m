datasetName = 'LastFM';
methodName = 'DSRPF';
%isSocial = 1;
%isSelfExciting = 0;
K = 20;
w = 1.0;

addpath('../Common/ReadData/');
addpath(genpath('../Common/Dynamic/'));

%% Reading Data
addressEvents = sprintf('../Datasets/%s.tsv',datasetName);
[U,P,events,eventsMatrix,userEvents] = readEventsInput(addressEvents);
addressEdges = sprintf('../Datasets/%s_adjList.txt',datasetName);
[inedges,outedges] = readEdges(addressEdges,U);

N = size(events,1);
t0 = zeros(U,1);
for u=1:U
    if (length(userEvents{u}) > 1)
        %go back from first event, with length of <second event-first event>
        t0(u) = userEvents{u}(1)-(userEvents{u}(2)-userEvents{u}(1));
    else
%         disp(u);
    end
end

trainSize = ceil(0.8*N);
trainEvents = events(1:trainSize);
testEvents = events(size(trainEvents)+1:end);

trainEventsMatrix = computeUserProductEventsMatrix(U,P,trainEvents);
testUserEventsMatrix = computeUserEventsMatrix(U,testEvents);
fprintf('Reading Dataset Completed.\n');
save(sprintf('../Datasets/%sDataset.mat',datasetName),'U','P','events','eventsMatrix','userEvents','trainEvents','testEvents','trainEventsMatrix','testUserEventsMatrix','inedges','outedges','t0');


%% initialization
prior = struct;
prior.shape = struct;
prior.shape.beta = 1.0;
prior.shape.theta = 1.0;
prior.shape.tau =1.0;
prior.shape.ksi = 1.0;
prior.shape.eta = 1.0;
prior.shape.mu = 1.0;

prior.rate = struct;
prior.rate.ksi = 0.1;
prior.rate.eta = 0.1;
prior.rate.mu = 0.1;

kernel = struct;
kernel.w = w;
kernel.g = @(x,w) w*exp(-w*x);
kernel.g_factorized = @(x,w,p) p*exp(-w*x);
kernel.g_log = @(x,w) log(w)-w*x;
kernel.G = @(x,w) 1-exp(-w*x) ;
kernel.F = @(i,j,T) integralOfProduct(i,j,T);

I = 31;
params = struct;
params.U = U;
params.P = P;
params.K = K;
params.I = I;
params.J = I;
params.maxNumberOfIterations = 10000;
params.saveInterval = 100;
params.plottingInIteration = 0;
params.datasetName = datasetName;
params.methodName = methodName;
params.phiComputationAlgorithm = 'linear';%'quadratic';

%% running algorithm
load(sprintf('../Datasets/%sDataset.mat',datasetName));

 [theta,beta,tau,gamma,prior,cuv,maeList,logLikelihoodList] = DynamicRPF(trainEvents,trainEventsMatrix,...
    trainEvents{end}.time,t0,inedges,outedges,prior,params,kernel);
fprintf('Learning Model Completed.\n');

%% metric evaluation
load(sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',...
    methodName,datasetName,params.K,w,params.maxNumberOfIterations));

% General Metrics
g = @(x) w*exp(-w*x);
RecListSize = 20;
[ranks, ndcgOverTime, ndcgAtK, recallAtKOverTime, recallAtK,...
   EstimatedReturningTime, RealReturningTime,diff] = ...
DynamicEvaluator(trainEvents,testEvents, outedges,inedges,eventsMatrix,...
   theta, beta, tau, g, params, RecListSize);

save(sprintf('../Results/Metrics_%s_%s_K_%d_w_%.1f_iter_%d.mat',methodName, datasetName,params.K,w,params.maxNumberOfIterations),...
   'ranks', 'ndcgOverTime', 'ndcgAtK', 'recallAtKOverTime', 'recallAtK',...
   'EstimatedReturningTime', 'RealReturningTime','diff');


%% plotting


%% remove path
rmpath('../Common/ReadData/');
rmpath(genpath('../Common/Dynamic/'));
