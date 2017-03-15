%day is in range[0-6]
%hour is in range[0-23]
function [day,hour] = dayAndHour(t)
    day = mod(floor(t/24),7);
    hour = mod(floor(t),24);
end