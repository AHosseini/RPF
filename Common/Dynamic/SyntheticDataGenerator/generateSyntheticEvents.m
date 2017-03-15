function [trainEvents,trainEventsMatrix,inedges,outedges,model] = ...
    generateSyntheticEvents(U,P,K,maxEvents,trainEventsSize)
%{
U = 10;
P = 5;
K = 3;
    %}
I = 31;

model = ModelGeneratorTemporalFeatures(U,P,K,I,0.005);
[events,eventsMatrix] = PurchaseSimulatorTemporalFeatures(model,maxEvents);
fprintf('model with %d events generated' , length(events));
outedges = cell(U,1); 
for u=1:U
    outedges{u} = find(model.tau{u});
end
inedges = cell(U,1);
for u=1:U
    inedges{u} = find(model.tauInvert{u});
end

t0 = zeros(U,1);
%{
testEvents = events;
testEventsMatrix = eventsMatrix;
%}
for trainSize=trainEventsSize
    trainEvents = events(1:trainSize);
    trainEventsMatrix = computeUserProductEventsMatrix(U,P,trainEvents);
    save(sprintf('../../Datasets/synthetic_events%d_U%d_P%d_K%d_Dataset',trainSize,U,P,K),...
        'inedges','outedges','trainEvents','trainEventsMatrix','U','P','K','I','t0','eventsMatrix','events','model');
end

end