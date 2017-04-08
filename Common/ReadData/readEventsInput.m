function [U,P,events,eventsMatrix,userEvents] = readEventsInput(address)
    eventsList = load(address);
    baseTime = eventsList(1,1)-(eventsList(2,1)-eventsList(1,1));
    eventsList(:,1) = eventsList(:,1)-baseTime;
    eventsList(:,1) = eventsList(:,1)/(60.0*60.0);
   
    events = cell(size(eventsList,1),1);
    U = max(eventsList(:,2));
    P = max(eventsList(:,3));
    eventsMatrix = cell(U,P);
    userEvents = cell(U,1);
    for i=1:size(eventsList,1)
        event = struct;
        event.time = eventsList(i,1);
        event.user = eventsList(i,2);
        event.product = eventsList(i,3);
        events{i} = event;
        eventsMatrix{event.user,event.product}(end+1) = event.time;
        userEvents{event.user}(end+1) = event.time;
    end
end