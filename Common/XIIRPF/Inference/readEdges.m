function [inedges,outedges] = readEdges(addressEdges,U)
    disp(U);
    a = zeros(U,U);
    fileID = fopen(addressEdges,'r');
    while (true)
        [u,cnt] = fscanf(fileID , '%d' , 1);
        
        if (cnt == 0)
            break;
        end
        
        neighbourSize = fscanf(fileID, '%d', 1);
        neighbours = fscanf(fileID , '%d', neighbourSize)';
        if (u <= U)
            %a(neighbours,u) = 1; %v in N(u) influences on u , (u is following v)
            for v=neighbours
                if (v <= U)
                    a(v,u) = 1;
                end
            end
            a(u,u) = 1;
        end
    end
    outedges = cell(U,1); 
    inedges = cell(U,1); 
    for u=1:U
        outedges{u} = find(a(u,:));
        inedges{u} = find(a(:,u)');
    end
end