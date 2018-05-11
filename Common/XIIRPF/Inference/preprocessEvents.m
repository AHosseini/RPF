function [ utriggers_times, utriggers_users, ptriggers_times, ptriggers_products]...
    = preprocessEvents( events, inedges, userEvents)
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
parfor n = 2:N
    un = events{n}.user;
    pn = events{n}.product;
    indices = userEvents{un}(userEvents{un}<n);
    utriggers_places = indices(product(indices)==pn);
    if sum(utriggers_places)>0
        utriggers_users{n} = user(utriggers_places);
        utriggers_times{n} = time(utriggers_places);
    end
    
    ptriggers_indices = indices(not(product(indices)==pn));
    if sum(ptriggers_indices)>0
        ptriggers_products{n} = product(ptriggers_indices);
        ptriggers_times{n} = time(ptriggers_indices);
    end
    if mod(n,100) ==0
        fprintf('iteration %d of %d for preprocessing data is completed.\n',n,N);
    end
%     ptriggers_count = 0;
%     for m = 1:n-1
%         tm = events{m}.time;
%         um = events{m}.user;
%         pm = events{m}.product;
%         if um==un && pm~=pn
%             ptriggers_count= ptriggers_count+1;
%             ptriggers_products_n(ptriggers_count) = pm;
%             ptriggers_times_n(ptriggers_count) = tm;
%         end
%         if sum(inedges{un}==um)>0 && pm==pn
%             utriggers_count = utriggers_count+1;
%             utriggers_times_n(utriggers_count) = tm;
%             utriggers_users_n(utriggers_count) = um;
%         end
%     end
%     if utriggers_count>0
%         utriggers_users{n}=utriggers_users_n(1:utriggers_count);
%         utriggers_times{n}=utriggers_times_n(1:utriggers_count);
%     end
%     if ptriggers_count>0
%         ptriggers_products{n}=ptriggers_products_n(1:ptriggers_count);
%         ptriggers_times{n}=ptriggers_times_n(1:ptriggers_count);
%     end
end
end