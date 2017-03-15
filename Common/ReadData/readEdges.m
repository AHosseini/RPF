function [inedges,outedges] = readEdges(addressEdges,U)
    disp(U);
    a = zeros(U,U);
    fileID = fopen(addressEdges,'r');
    while (true)
        [u,cnt] = fscanf(fileID , '%d' , 1);
        
        if (cnt == 0)
            break;
        end
        
%         if (userCountMap.isKey(u) == 1)        
%             u = userCountMap(u);
%         else
%             disp('error u not found');
%             disp(u);
%             return;
%         end
        
        neighbourSize = fscanf(fileID, '%d', 1);
        neighbours = fscanf(fileID , '%d', neighbourSize)';
%         for i=1:length(neighbours)
%             v = neighbours(i);
%             if (userCountMap.isKey(v) == 1)
%                 neighbours(i) = userCountMap(v);
%             else
%                 disp('error v not found');
%                 disp(v);
%                 return;
%             end
%         end
        
        a(neighbours,u) = 1; %v in N(u) influences on u , (u is following v)
        a(u,u) = 1;
        
        %{        
        disp(u);
        disp(neighbours);
         disp('---------------------------------------------------------------');
        if (u > 5)
            break;
        end
        %}
    end
    outedges = cell(U,1); 
    inedges = cell(U,1); 
    for u=1:U
        outedges{u} = find(a(u,:));
        inedges{u} = find(a(:,u)');
    end
end