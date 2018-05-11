function [eventsMatrix] = computeUserProductEventsMatrix(U,P,events)
    eventsMatrix = cell(U,P);
    for i=1:size(events)
        u = events{i}.user;
        p = events{i}.product;
        t = events{i}.time;
        eventsMatrix{u,p}(end+1) = t;
    end
end