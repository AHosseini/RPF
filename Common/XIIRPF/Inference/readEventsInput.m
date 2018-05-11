function [U,P,B,events,eventsMatrix, itemsCluster,itemsSimilarity] = readEventsInput(address,maxU,maxP)
    if nargin<3
        maxP=inf;
    end
    if nargin<2
        maxU = inf;
    end
    events = cell(1,1);
    load(address);
    eventsList = events;
    times = eventsList(:,1);
    users = eventsList(:,2);
    products = eventsList(:,3);
    U = min(max(eventsList(:,2)),maxU);
    P = min(max(eventsList(:,3)),maxP);
    B = 22;
    itemsCluster = zeros(P,1);
    itemsSimilarity = zeros(P,P);
    for p = 1:P
%         fprintf('size(find(features(p,1386:1407)))=[%d,%d]\n',size(find(features(p,1386:1407))));
        itemsCluster(p) = find(features(p,1416:1420));
        for q = 1:p-1
            featurep = features(p,1416:1420);%[features(p,1:1386),features(p,1408:1420)];
            featureq = features(q,1416:1420);%[features(q,1:1386),features(q,1408:1420)];
            itemsSimilarity(p,q) = featurep*featureq';
            itemsSimilarity(q,p) = itemsSimilarity(p,q);
        end
    end
    events = cell(size(eventsList,1),1);
    eventsMatrix = cell(U,P);
    cnt = 0;
    for i=1:size(eventsList,1)
        event = struct;
        event.time = times(i);
        event.user = users(i);
        event.product = products(i);
        if (event.user <= maxU && event.product <= maxP)
            cnt = cnt+1;
            events{cnt} = event;
            eventsMatrix{event.user,event.product}(end+1) = event.time;
        end
    end
end