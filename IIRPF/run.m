%% These parameters are set based on the dataset and can be changed
datasetName = 'LastFM';
methodName = 'IIRPF';
isSocial = 1;
isSelfExciting = 1;
forceRun = 1;
K = 20;
w = 1.0;
startIteration = 0;
%% Adding path
addpath('../Common/ReadData/');
addpath(genpath('../Common/IIRPF/'));

%% Reading data
if exist(sprintf('../../../Datasets/%sDataset_%d_%d.mat',datasetName,isSocial,isSelfExciting),'file')~=2
    fprintf('Dataset does not exist. Going to build dataset.\n');
    addressEvents = sprintf('../../../../RawDatasets/%s.tsv',datasetName);
    [U,P,events,eventsMatrix,~] = readEventsInput(addressEvents); 
    inedges = cell(U,1);
    outedges = cell(U,1);
    if isSocial
        addressEdges = sprintf('../../../../RawDatasets/%s_adjList.txt',datasetName);
        [inedges,outedges] = readEdges(addressEdges,U);
        if isSelfExciting
            for u = 1:U
                isInList = 0;
                temp = inedges{u};
                for i = 1:length(inedges{u})
                    if temp(i)==u
                        isInList = 1;
                    end
                end
                if isInList==0
                    inedges{u}=[u,inedges{u}];
                    outedges{u} = [u,outedges{u}];
                end
            end
        end
    else
         for u =1:U
            if isSelfExciting
                inedges{u} = u;
                outedges{u} = u;
            else
                inedges{u} = [];
                outedges{u} = [];
            end
        end
    end
    fprintf('Building Datasets Completed.\n');
    save(sprintf('../../../Datasets/%sDataset_%d_%d',datasetName,isSocial,isSelfExciting),'U','P','events',...
    'eventsMatrix','inedges','outedges');
else
    load(sprintf('../../../Datasets/%sDataset_%d_%d',datasetName,isSocial,isSelfExciting));
end
N = size(events,1);
t0 = zeros(U,1);
trainSize = ceil(0.8*N);
trainEvents = events(1:trainSize);
testEvents = events(size(trainEvents)+1:end);

trainEventsMatrix = computeUserProductEventsMatrix(U,P,trainEvents);
trinUserEvents = computeUserEventsMatrix(U,trainEvents);
fprintf('Reading Dataset Completed.\n');
%% Initialization
prior = struct;
prior.shape = struct;
prior.shape.beta = 1.0;
prior.shape.theta = 1.0;
prior.shape.tau = 1.0;
prior.shape.phi = 1.0;

prior.shape.ksi = 1.0;
prior.shape.eta = 1.0;
prior.shape.mu = 1.0;
prior.shape.psi = 1.0;


prior.rate = struct;
prior.rate.ksi = 0.1;
prior.rate.eta = 0.1;
prior.rate.mu = 0.1;
prior.rate.psi = 0.1;

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
params.maxNumberOfIterations = 1000;
params.saveInterval = 50;
params.plottingInIteration = 0;
params.datasetName = datasetName;
params.methodName = methodName;
%% Training The Model
modelFileName = sprintf('../../../LearnedModels/LearnedModel_%s_%s_K_%d_iter_%d.mat',params.methodName,...
    datasetName,params.K,params.maxNumberOfIterations);
if forceRun==1 || exist(modelFileName,'file')~=2
    [theta,beta,tau,phi, gamma, prior, cuv] = IIRPF(trainEvents,trainEventsMatrix,trinUserEvents,...
        events{end}.time,t0,inedges,outedges,prior,params,kernel,0,startIteration);
    save(modelFileName,'theta','beta','tau','phi','kernel','params','prior','gamma','cuv');
    fprintf('Learning Model Completed.\n');
end
fprintf('Learning Model Completed.\n');

%% evaluation
modelFileName = sprintf('../../../LearnedModels/LearnedModel_%s_%s_K_%d_iter_%d.mat',params.methodName,...
    datasetName,params.K,params.maxNumberOfIterations);
load(modelFileName);
g = @(x) w*exp(-w*x);
RecListSize = 20;
[ranks, ndcgOverTime, ndcgAtK, recallAtKOverTime, recallAtK,...
     EstimatedReturningTime, RealReturningTime,diff] = ...
Evaluator(trainEvents,testEvents, outedges,inedges,eventsMatrix,...
    theta, beta, tau, phi, g, params, RecListSize);
save(sprintf('../../../Results/Metrics_%s_%s_K_%d_w_%.1f_iter_%d',methodName,...
        datasetName,params.K,w,params.maxNumberOfIterations),...
    'ranks', 'ndcgOverTime', 'ndcgAtK', 'recallAtKOverTime', 'recallAtK',...
    'EstimatedReturningTime', 'RealReturningTime','diff');
fprintf('Evaluation Completed.\n');
%% QQ plot Data
g = @(x,w) w*exp(-w*x);
w=1;
RecListSize = 20;
InterEventIntensityIntegrals = computeInterEventIntensityIntegral(events,theta, beta, tau, phi, outedges, w, g, params);
save(sprintf('../../../Results/InterEventIntensityIntegrals_%s_%s_K_%d_w_%.1f_iter_%d',methodName, datasetName,params.K,w,params.maxNumberOfIterations),...
    'InterEventIntensityIntegrals');
%% remove path
rmpath('../Common/ReadData/');
rmpath(genpath('../Common/NonDynamic/'));