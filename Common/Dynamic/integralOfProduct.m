%% integral of product
function [integral] = integralOfProduct(i,j,T)
    if (1 <= i && i <= 7)
        if (i == j)
            integral = integralDay(i-1,T);
        elseif (1 <= j && j <= 7)
            integral = 0;
        else
            integral = integralDayAndHour(i-1,j-8,T);
        end
    else %8 <= i <= 31 --> i-8 = hour
        if (i == j)
            integral = integralHour(i-8,T);
        elseif (8 <= j && j <= 31)
            integral = 0.0;
        else
            integral = integralDayAndHour(j-1,i-8,T);
        end
    end    
end

function [integral] = integralHour(h,T)
    [day,hour] = dayAndHour(T);
    partial = 0.0;
    if (h < hour)
        partial = 1.0;
    elseif (h == hour)
        partial = (T-floor(T)*1.0);
    end
    
    numberOfDays = floor(T/24);
    integral = numberOfDays*1.0+partial;
end

function [integral] = integralDay(d,T)
    [day,hour] = dayAndHour(T);
    partial = 0.0;
    if (d < day)
        partial = 24.0;
    elseif (d == day)
        partial = (T-floor(T/24)*24);
    end    
    numberOfDays = floor(T/24);
    numberOfWeeks = floor(numberOfDays/7);
    integral = numberOfWeeks*24.0+partial;
end

function [integral] = integralDayAndHour(d,h,T)
    [day,hour] = dayAndHour(T);
    partial = 0.0;
    if ((d < day) || (d == day && h < hour))
        partial = 1.0;
    elseif (d == day && h == hour)
        partial = (T-floor(T)*1.0);
    end
    numberOfDays = floor(T/24);
    numberOfWeeks = floor(numberOfDays/7);
    integral = numberOfWeeks*1.0+partial;
end