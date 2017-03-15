%% Defining Colors
HPFColor=[239,44,123]/255;
SPFColor=[91,181,231]/255;
DPFColor = [22,157,116]/255;
TSRSColor = [185,41,35]/255;
HRPFColor=[0,84,166]/255;
SRPFColor=[211,94,26]/255;
DRPFColor=[210,181,51]/255;
DSRPFColor=[90,41,7]/255;
%% Loading Data
datasetName = 'lastfm';
% datasetName = 'tianchi';
hpf = load(sprintf('%s/hpf_%s_Metrics.mat',datasetName,datasetName));
spf = load(sprintf('%s/spf_%s_Metrics.mat',datasetName,datasetName));
dpf = load(sprintf('%s/dpf_%s_Metrics.mat',datasetName,datasetName));

TSRS = load(sprintf('%s/TSRS_Metrics_%s_K_20_iter_1000.mat',datasetName,datasetName));

HRPF = load(sprintf('%s/Metrics_HRPF_%s_K_20_w_1.0_iter_1000.mat',datasetName,datasetName));
SRPF = load(sprintf('%s/Metrics_SRPF_%s_K_20_w_1.0_iter_1000.mat',datasetName,datasetName));
DRPF = load(sprintf('%s/Metrics_DRPF_%s_K_20_w_1.0_iter_1000.mat',datasetName,datasetName));
DSRPF = load(sprintf('%s/Metrics_DSRPF_%s_K_20_w_1.0_iter_1000.mat',datasetName,datasetName));


%% Load Dataset 
load(sprintf('../Datasets/%sDataset.mat',datasetName));
NTest =length(testEvents);
testTimes = zeros(1,NTest);
for i = 1:NTest
    testTimes(i) = testEvents{i}.time;
end
% TimeSamplingInterval = 100;
%% NDCG Over Time
N = length(TSRS.ndcgOverTime);
% Create figure
figure1 = figure('InvertHardcopy','off','Color',[1 1 1]);

% Create axes
axes1 = axes('Parent',figure1,'LineWidth',2,'FontWeight','bold',...
    'FontName','times new roman');
hold(axes1,'on');
start = 100;
TimeSamplingInterval = 1;
plot(testTimes(1:TimeSamplingInterval:N)*3600,hpf.ndcgOverTime(1:TimeSamplingInterval:N),'Color',HPFColor,'LineWidth',2,'Parent',axes1);
plot(testTimes(start:TimeSamplingInterval:N)*3600,spf.ndcgOverTime(start:TimeSamplingInterval:N),'Color',SPFColor,'LineWidth',2,'Parent',axes1);
plot(testTimes(start:TimeSamplingInterval:N)*3600,dpf.ndcgOverTime(start:TimeSamplingInterval:N),'Color',DPFColor,'LineWidth',2,'Parent',axes1);
plot(testTimes(start:TimeSamplingInterval:N)*3600,TSRS.ndcgOverTime(start:TimeSamplingInterval:N),'Color',TSRSColor,'LineWidth',2,'Parent',axes1);
plot(testTimes(start:TimeSamplingInterval:N)*3600,HRPF.ndcgOverTime(start:TimeSamplingInterval:N),'Color',HRPFColor,'LineWidth',2,'Parent',axes1);
plot(testTimes(start:TimeSamplingInterval:N)*3600,SRPF.ndcgOverTime(start:TimeSamplingInterval:N),'Color',SRPFColor,'LineWidth',2,'Parent',axes1);
plot(testTimes(start:TimeSamplingInterval:N)*3600,DRPF.ndcgOverTime(start:TimeSamplingInterval:N),'Color',DRPFColor,'LineWidth',2,'Parent',axes1);
plot(testTimes(start:TimeSamplingInterval:N)*3600,DSRPF.ndcgOverTime(start:TimeSamplingInterval:N),'Color',DSRPFColor,'LineWidth',2,'Parent',axes1);
ylabel('NDCG','FontWeight','bold','FontSize',24,...
    'FontName','times new roman');
xlabel('Time','FontWeight','bold','FontSize',24,'FontName','times new roman');
% Create legend
legend('HPF','SPF','DPF','TSRS','HRPF','SRPF','DRPF','DSRPF');
% legend('HPF','DPF','TSRS','HRPF','DRPF');
legend1 = legend(axes1,'show','Location','northwest');
set(legend1,'FontSize',20);
grid
% xlim(axes1,[1 N]);
ylim(axes1,[0 1]);
box(axes1,'on');
%% NDCG@K
% Create figure
figure2 = figure('InvertHardcopy','off','Color',[1 1 1]);

% Create axes
axes2 = axes('Parent',figure2,'LineWidth',2,'FontWeight','bold',...
    'FontName','times new roman');
hold(axes2,'on');
plot(hpf.ndcgAtK,'Color',HPFColor,'LineWidth',2,'Parent',axes2);
% plot(spf.ndcgAtK,'Color',SPFColor,'LineWidth',2,'Parent',axes2);
plot(dpf.ndcgAtK,'Color',DPFColor,'LineWidth',2,'Parent',axes2);
plot(TSRS.ndcgAtK,'Color',TSRSColor,'LineWidth',2,'Parent',axes2);
plot(HRPF.ndcgAtK,'Color',HRPFColor,'LineWidth',2,'Parent',axes2);
% plot(SRPF.ndcgAtK,'Color',SRPFColor,'LineWidth',2,'Parent',axes2);
plot(DRPF.ndcgAtK,'Color',DRPFColor,'LineWidth',2,'Parent',axes2);
% plot(DSRPF.ndcgAtK,'Color',DSRPFColor,'LineWidth',2,'Parent',axes2);
% legend('HPF','SRPF','DRPF','TSRS','HRPF','SRPF','DRPF','DSRPF');
legend('HPF','DPF','TSRS','HRPF','DRPF');
ylabel('ndcg@K','FontWeight','bold','FontSize',24,...
    'FontName','times new roman');
xlabel('K','FontWeight','bold','FontSize',24,'FontName','times new roman');
legend2 = legend(axes2,'show');
set(legend2,'FontSize',20);
grid
xlim(axes2,[1 30]);
ylim(axes2,[0 1]);
box(axes2,'on');
%% Recall@K
% Create figure
% figure3 = figure('InvertHardcopy','off','Color',[1 1 1]);
% 
% % Create axes
% axes3 = axes('Parent',figure3,'LineWidth',2,'FontWeight','bold',...
%     'FontName','times new roman');
% hold(axes3,'on');
% plot(TSRS.recallAtK,'Color',TSRSColor,'LineWidth',2,'Parent',axes3);
% plot(NS_RPF.recallAtK,'Color',NSRPFColor,'LineWidth',2,'Parent',axes3);
% plot(ND_RPF.recallAtK,'Color',NDRPFColor,'LineWidth',2,'Parent',axes3);
% plot(RPF.recallAtK,'Color',RPFColor,'LineWidth',2,'Parent',axes3);
% ylabel('Recall@K','FontWeight','bold','FontSize',13,...
%     'FontName','times new roman');
% xlabel('K','FontWeight','bold','FontSize',13,...
%     'FontName','times new roman');
% legend('TSRS','NS-RPF','ND-RPF','RPF');
% legend3 = legend(axes3,'show');
% set(legend3,'FontSize',12,'Location','best');
% grid
% xlim(axes3,[1 30]);
%% RecallAt20 Over Time
% Create figure
% figure4 = figure('InvertHardcopy','off','Color',[1 1 1]);
% 
% % Create axes
% axes4 = axes('Parent',figure4,'LineWidth',2,'FontWeight','bold',...
%     'FontName','times new roman');
% hold(axes4,'on');
% N = length(TSRS.recallAtKOverTime);
% plot(1:TimeSamplingInterval:N,TSRS.recallAtKOverTime(1:TimeSamplingInterval:N),'Color',TSRSColor,'LineWidth',2,'Parent',axes4);
% plot(1:TimeSamplingInterval:N,NS_RPF.recallAtKOverTime(1:TimeSamplingInterval:N),'Color',NSRPFColor,'LineWidth',2,'Parent',axes4);
% plot(1:TimeSamplingInterval:N,ND_RPF.recallAtKOverTime(1:TimeSamplingInterval:N),'Color',NDRPFColor,'LineWidth',2,'Parent',axes4);
% plot(1:TimeSamplingInterval:N,RPF.recallAtKOverTime(1:TimeSamplingInterval:N),'Color',RPFColor,'LineWidth',2,'Parent',axes4);
% ylabel('recall@20 OverTime','FontWeight','bold','FontSize',13,...
%     'FontName','times new roman');
% xlabel('Event Number','FontWeight','bold','FontSize',13,...
%     'FontName','times new roman');
% legend('TSRS','NS-RPF','ND-RPF','RPF');
% legend4 = legend(axes4,'show');
% set(legend4,'FontSize',12,'Location','best');
% grid
% xlim(axes4,[1 N]);

%% QQ Plot
% NTrain = 320*1000;
% SamplingRate = 10;
DRPFLastfmInterEventTimes = load('InterEventIntensityIntegrals_DRPF_lastfm_K_20_w_1.0_iter_1000.mat');
DRPFLastfmInterEventTimes = DRPFLastfmInterEventTimes.InterEventIntensityIntegrals;
DRPFLastfmInterEventTimes = DRPFLastfmInterEventTimes(DRPFLastfmInterEventTimes>40)/40;
% DRPFLastfmInterEventTimes = DRPFLastfmInterEventTimes(1:SamplingRate:end);
csvwrite('DRPF-lastfm.csv',DRPFLastfmInterEventTimes);
DSRPFLastfmInterEventTimes = load('InterEventIntensityIntegrals_DSRPF_lastfm_K_20_w_1.0_iter_1000.mat');
DSRPFLastfmInterEventTimes = DSRPFLastfmInterEventTimes.InterEventIntensityIntegrals;
DSRPFLastfmInterEventTimes = DSRPFLastfmInterEventTimes(DSRPFLastfmInterEventTimes>40)/40;
% DSRPFLastfmInterEventTimes = DSRPFLastfmInterEventTimes(1:SamplingRate:end)/40;
csvwrite('DSRPF-lastfm.csv',DSRPFLastfmInterEventTimes);
HRPFLastfmInterEventTimes = load('InterEventsIntegral_HRPF_lastfm_K_20_w_1.0_iter_1000.mat');
HRPFLastfmInterEventTimes = HRPFLastfmInterEventTimes.InterEventIntensityIntegrals;
HRPFLastfmInterEventTimes = HRPFLastfmInterEventTimes(HRPFLastfmInterEventTimes>40)/40;
% HRPFLastfmInterEventTimes = HRPFLastfmInterEventTimes(1:SamplingRate:end)/40;
csvwrite('HRPF-lastfm.csv',HRPFLastfmInterEventTimes);
SRPFLastfmInterEventTimes = load('InterEventsIntegral_SRPF_lastfm_K_20_w_1.0_iter_1000.mat');
SRPFLastfmInterEventTimes = SRPFLastfmInterEventTimes.InterEventIntensityIntegrals;
SRPFLastfmInterEventTimes = SRPFLastfmInterEventTimes(SRPFLastfmInterEventTimes>40)/40;
% SRPFLastfmInterEventTimes = SRPFLastfmInterEventTimes(1:SamplingRate:end)/40;
csvwrite('SRPF-lastfm.csv',SRPFLastfmInterEventTimes);
hold on;
% qqplot(DRPFLastfmInterEventTimes,makedist('Exponential'));
% qqplot(DSRPFLastfmInterEventTimes,makedist('Exponential'));
qqplot(HRPFLastfmInterEventTimes,makedist('Exponential'));
qqplot(SRPFLastfmInterEventTimes,makedist('Exponential'));
legend('DRPF','DSRPF','HRPF','SRPF');
%%
%Tianchi
NTrain = 800*1000;
SamplingRate = 10;
DRPFTianchiInterEventTimes = load('InterEventsIntegral_DRPF_tianchi_K_20_w_1.0_iter_1000.mat');
DRPFTianchiInterEventTimes = DRPFTianchiInterEventTimes.InterEventIntensityIntegrals(1:SamplingRate:NTrain);
HRPFTianchiInterEventTimes = load('InterEventsIntegral_HRPF_tianchi_K_20_w_1.0_iter_1000.mat');
HRPFTianchiInterEventTimes = HRPFTianchiInterEventTimes.InterEventIntensityIntegrals(1:SamplingRate:NTrain);
figure('name','tianchi QQ-plot');
qqplot(DRPFTianchiInterEventTimes,makedist('Exponential'));
qqplot(HRPFTianchiInterEventTimes,makedist('Exponential'));
legend('DRPF','HRPF');
%%
%Synthetic
NTrain = 800*1000;
SamplingRate = 100;
DSRPFSyntheticInterEventTimes = load('InterEventsIntegral_DS-RPF_synthetic_events1000000_U1000_P1000_K10__K_10_w_1.0_iter_10000.mat');
DSRPFSyntheticInterEventTimes = DSRPFSyntheticInterEventTimes.InterEventIntensityIntegrals(1:SamplingRate:NTrain);
figure('name','synthetic');
qqplot(DSRPFSyntheticInterEventTimes,makedist('Exponential'));


%% 
for i = 1:NTrain
    
end