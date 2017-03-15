function [userEventsMatrix] = computeUserEventsMatrix(U,events)
    userEventsMatrix = cell(U,1);
    for i=1:size(events)
        u = events{i}.user;
        userEventsMatrix{u}(end+1) = events{i};
    end
end
