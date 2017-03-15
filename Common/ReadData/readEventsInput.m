function [U,P,events,eventsMatrix,userEvents] = readEventsInput(address)
    eventsList = load(address);
    baseTime = eventsList(1,1)-(eventsList(2,1)-eventsList(1,1));
    eventsList(:,1) = eventsList(:,1)-baseTime;
    eventsList(:,1) = eventsList(:,1)/(60.0*60.0);
    
%     userCountMap = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
%     productCountMap = containers.Map('KeyType', 'int32', 'ValueType', 'int32');
    
    events = cell(size(eventsList,1),1);
    U = max(eventsList(:,2));
    P = max(eventsList(:,3));
    eventsMatrix = cell(U,P);
    userEvents = cell(U,1);
    for i=1:size(eventsList,1)
%         u = eventsList(i,2);
%         if (userCountMap.isKey(u) == 1)
%             u = userCountMap(u);    
%         else
%             userCountMap(u) = userCountMap.Count+1;
%             u = userCountMap.Count;
%         end
%         eventsList(i,2) = u;
        
%         p = eventsList(i,3);
%         p = floor(p); %TODO : remove this line
        
%         if (productCountMap.isKey(p) == 1)
%             p = productCountMap(p);
%         else
%             productCountMap(p) = productCountMap.Count+1;
%             p = productCountMap.Count;
%         end
%         eventsList(i,3) = p;
        event = struct;
        event.time = eventsList(i,1);
        event.user = eventsList(i,2);
        event.product = eventsList(i,3);
        events{i} = event;
        eventsMatrix{event.user,event.product}(end+1) = event.time;
        userEvents{event.user}(end+1) = event.time;
    end
    
%     U = double(userCountMap.Count);
%     P = productCountMap.Count;
%     disp(U);
%     disp(P);
    
%     eventsMatrix = computeUserProductEventsMatrix(U,P,events);
    
    
%    disp(size(eventsList));
end