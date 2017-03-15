function [u,p] = sample(Is)
    u = sample1d(sum(Is,2));
    p = sample1d(Is(u,:));
end

function d = sample1d(Is)
    %disp(Is);
    u = rand()*sum(Is);
    %disp(['sum(Is) = ',num2str(sum(Is)),' u = ',num2str(u)]);
    sumI = 0;
    for d=1:length(Is)
        sumI = sumI+Is(d);
        if (u <= sumI)
            break;
        end
    end
end