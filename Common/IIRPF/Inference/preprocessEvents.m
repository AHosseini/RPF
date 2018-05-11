function [ utriggers_times, utriggers_users,ptriggers_times, ptriggers_products]...
    = preprocessEvents( events, inedges)
%This function preprocesses the event to find the probable triggering
%events for each event
N = length(events);
user = zeros(N,1);
time = zeros(N,1);
product = zeros(N,1);
for n = 1:N
    time(n) = events{n}.time;
    user(n) = events{n}.user;
    product(n) = events{n}.product;
end
utriggers_times = cell(N,1);
utriggers_users = cell(N,1);
ptriggers_times = cell(N,1);
ptriggers_products = cell(N,1);
for n = 2:N
    un = events{n}.user;
    pn = events{n}.product;
    indices = 1:n-1;
    utriggers_places = product(indices)==pn;
    utriggers_indices = find(utriggers_places);
    for i=1:length(utriggers_indices)
        index = utriggers_indices(i);
        if sum(inedges{un}==user(index))==0
            utriggers_places(index) = false;
        end
    end
    if sum(utriggers_places)>0
        utriggers_users{n} = user(utriggers_places);
        utriggers_times{n} = time(utriggers_places);
    end
    
    ptriggers_indices = user(indices)==un & not(product(indices)==pn);
    if sum(ptriggers_indices)>0
        ptriggers_products{n} = product(ptriggers_indices);
        ptriggers_times{n} = time(ptriggers_indices);
    end
    if mod(n,100) ==0
        fprintf('iteration %d of %d for preprocessing data is completed.\n',n,N);
    end
end
end