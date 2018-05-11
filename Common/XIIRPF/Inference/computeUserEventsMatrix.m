function [userEvents] = computeUserEventsMatrix(U,events)
    userEvents = cell(U,1);
    for u = 1:U
        userEvents{u} = struct;
        userEvents{u}.time = [];
        userEvents{u}.product = [];
    end
    for i=1:size(events)
        u = events{i}.user;
        userEvents{u}.time(end+1) = events{i}.time;
        userEvents{u}.product(end+1) = events{i}.product;
    end
end
