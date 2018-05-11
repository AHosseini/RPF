datasetName = 'LastFM';
methodName = 'HRPF';
%isSocial = 0;
%isSelfExciting = 1;
K = 20;
w = 1.0;

addpath('../Common/ReadData/');
addpath(genpath('../Common/NonDynamic/'));

%% Reading data
addressEvents = sprintf('../Datasets/%s.tsv',datasetName);
[U,P,events,eventsMatrix,userEvents] = readEventsInput(addressEvents);

outedges = cell(U,1); 
inedges = cell(U,1); 
for u=1:U
    inedges{u} = u;
    outedges{u} = u;
end

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


%% Initialization
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

params = struct;
params.U = U;
params.P = P;
params.K = K;
params.maxNumberOfIterations = 10000;
params.saveInterval = 100;
params.plottingInIteration = 0;
params.datasetName = datasetName;
params.methodName = methodName;
params.phiComputationAlgorithm = 'quadratic';%'linear';

%% running algorithm
load(sprintf('../Datasets/%sDataset.mat',datasetName));

[theta,beta,tau,gamma,prior,cuv] = NonDynamicRPF(trainEvents,trainEventsMatrix,...
    events{end}.time,t0,inedges,outedges,prior,params,kernel);
fprintf('Learning Model Completed.\n');

%% Metric Generation
load(sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w_%.1f_iter_%d.mat',...
    methodName,datasetName,params.K,w,params.maxNumberOfIterations));

% General Metrics
g = @(x) w*exp(-w*x);
RecListSize = 20;

[ranks, ndcgOverTime, ndcgAtK, recallAtKOverTime, recallAtK,...
    EstimatedReturningTime, RealReturningTime,diff] = ...
NonDynamicEvaluator(trainEvents,testEvents, outedges,inedges,eventsMatrix,...
    theta, beta, tau, g, params, RecListSize);
save(sprintf('../Results/Metrics_%s_%s_K_%d_w_%.1f_iter_%d',methodName, datasetName,params.K,w,params.maxNumberOfIterations),...
    'ranks', 'ndcgOverTime', 'ndcgAtK', 'recallAtKOverTime', 'recallAtK',...
    'EstimatedReturningTime', 'RealReturningTime','diff');
%% remove path
rmpath('../Common/ReadData/');
rmpath(genpath('../Common/NonDynamic/'));