%% These parameters are set based on the dataset and can be changed
datasetName = 'IPTV';
methodName = 'XIIRPF';
isSocial = 0;
isSelfExciting = 1;
forceRun = 1;
K = 20;
w2 = 0.1;
w1 = 0.1;
nu = 1;
K = 20;
startIteration = 0;
maxNumberOfIterations = 400;
%% Adding path
addpath(genpath('../Common/XIIRPF/'));

%% Reading data
if exist(sprintf('../Datasets/%sDataset_%d_%d.mat',datasetName,isSocial,isSelfExciting),'file')~=2
    fprintf('Dataset does not exist. Going to build dataset.\n');
    addressEvents = sprintf('../RawDatasets/%s.mat',datasetName);
    [U,P,B,events,eventsMatrix, itemsCluster,itemsSimilarity] = readEventsInput(addressEvents); 
    inedges = cell(U,1);
    outedges = cell(U,1);
    if isSocial
        addressEdges = sprintf('../RawDatasets/%s_adjList.txt',datasetName);
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
    fprintf('Building Datasets Completed. Going to save the dataset.\n');
    fprintf('Saving Dataset Completed.\n');
else
    fprintf('Dataset Already exists.');
    load(sprintf('../Datasets/%sDataset_%d_%d',datasetName,isSocial,isSelfExciting));
    fprintf('Dataset loaded.\n');
end
N = size(events,1);
t0 = zeros(U,1);
% train and test events
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
prior.shape.pi = 1.0;

prior.shape.ksi = 1.0;
prior.shape.eta = 1.0;
prior.shape.mu = 1.0;
prior.shape.psi = 1.0;
prior.shape.rho = 1.0;


prior.rate = struct;
prior.rate.ksi = 0.1;
prior.rate.eta = 0.1;
prior.rate.mu = 0.1;
prior.rate.psi = 0.1;
prior.rate.rho = 0.1;

kernel = struct;
kernel.w1 = w1;
kernel.w2 = w2;
kernel.g = @(x,w) exp(-w*x);
kernel.g_log = @(x,w) -w*x;
kernel.G = @(x,w) 1/w*(1-exp(-w*x));

kernel.nu = nu;
kernel.d = @(dist) (dist)/kernel.nu;
kernel.d_log = @(dist) log(dist)-log(kernel.nu);

params = struct;
params.U = U;
params.P = P;
params.K = K;
params.B = B;
params.maxNumberOfIterations = maxNumberOfIterations;
params.startIteration = startIteration;
params.saveInterval = 100;
params.plottingInIteration = 0;
params.datasetName = datasetName;
params.methodName = methodName;
%% Training The Model
modelFileName = sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',...
    methodName,datasetName,params.K,kernel.w1,kernel.w2,params.maxNumberOfIterations);
if forceRun==1 || exist(modelFileName,'file')~=2
    [expected_theta,expected_beta,expected_tau,expected_phi, ...
    expectedPi, gamma,prior,cuv,cbb] = XIIRPF(...
        trainEvents,trainEventsMatrix,trinUserEvents,itemsCluster,itemsSimilarity,...
        events{end}.time,t0,inedges,outedges,prior,params,kernel);
    save(modelFileName, 'expected_theta', 'expected_beta', 'expected_tau', 'expected_phi', 'expectedPi', 'kernel', ...
        'params','prior','gamma','cuv','cbb');
    fprintf('Learning Model Completed.\n');
end
fprintf('Learning Model Completed.\n');

%% evaluation
modelFileName = sprintf('../LearnedModels/LearnedModel_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',...
    methodName,datasetName,params.K,kernel.w1,kernel.w2,params.maxNumberOfIterations);
metricFileName = sprintf('../Results/Metrics_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',methodName,...
        datasetName,params.K,kernel.w1,kernel.w2,params.maxNumberOfIterations);
load(modelFileName);
if exist(metricFileName,'file')~=2
    RecListSize = 20;
    [ranks, ndcgOverTime, ndcgAtK, recallAtKOverTime, recallAtK,...
         EstimatedReturningTime, RealReturningTime,diff] = ...
    Evaluate(trainEvents,testEvents, outedges,inedges,eventsMatrix,itemsCluster,itemsSimilarity,...
        expected_theta, expected_beta, expected_tau, expected_phi, expectedPi, kernel, params, RecListSize);
    save(sprintf('../Results/Metrics_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',methodName,...
            datasetName,params.K,kernel.w1,kernel.w2,params.maxNumberOfIterations),...
        'ranks', 'ndcgOverTime', 'ndcgAtK', 'recallAtKOverTime', 'recallAtK',...
        'EstimatedReturningTime', 'RealReturningTime','diff');
end
fprintf('Evaluation Completed.\n');
%% QQ plot Data
InterEventIntensityIntegrals = computeInterEventIntensityIntegral(events,itemsCluster,itemsSimilarity,expected_theta, expected_beta, expected_tau, expected_phi, expectedPi, outedges, kernel, params);
save(sprintf('../Results/InterEventIntensityIntegrals_%s_%s_K_%d_w1_%.1f_w2_%.1f_iter_%d.mat',...
    methodName, datasetName,params.K,kernel.w1,kernel.w2,params.maxNumberOfIterations),...
    'InterEventIntensityIntegrals');