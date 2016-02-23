%   Normalized Cuts and Image Segmentation
%   Author : Satyabrat Srikumar
%            IIIT Bangalore
% ------------------------------------------------------------------------%          

function Ncuts
I = imread('test.jpg');
[no_rows, no_cols, c] = size(I);
N = no_rows * no_cols;


%-----------------------Parameter speicifications ------------------------%
r = 2;
sigma_I =4;
sigma_X =6;
threshold_Ncut = 0.16;
threshold_Area = 100;

%-------------------------------------------------------------------------%
%V_node - denotes all pixels as nodes of a graph 

V_node = zeros(N,c);
for k = 1:c
    temp = 1;
    for i = 1:no_cols
      for j = 1:no_rows
          V_node(temp,k) = I(j,i,k);
          temp = temp + 1;        
      end
    end 
end
 
%---------------------------SIMILARITY MATRIX CREATION-----------------------------%
%W - similarity matrix
W = sparse(N,N);  


% X - Spatial location matrix 
X_temp = zeros(no_rows, no_cols, 2);
for i = 1:no_rows
    for j = 1:no_cols
        X_temp(i,j,1) = i;
        X_temp(i,j,2) = j;
    end
end

X = zeros(N,1,2);
for k = 1:2
      temp = 1;
      for i = 1:no_cols
        for j = 1:no_rows
            X(temp,1,k) = X_temp(j,i,k);
            temp = temp + 1;        
        end
      end 
end

%F - Intensity Feature vectors
F = zeros(N,1,c);
for k = 1:c
   temp = 1;
   for i = 1:no_cols
       for j = 1:no_rows
           F(temp,1,k) = I(j,i,k);     
           temp = temp + 1;        
       end
   end 
end
F = uint8(F); %uint class required for addition compatibility with spatial
              %location matrix.  

% main loop 
r1 = floor(r);
for m =1:no_cols
    for n =1:no_rows
        
        %satisfies X(j)-r < X(i) < X(j)+r  
        range_cols = (m - r1) : (m + r1); 
        range_rows = ((n - r1) :(n + r1))';
        v_col_index = range_cols >= 1 & range_cols <= no_cols;  %valid col. index
        v_row_index = range_rows >= 1 & range_rows <= no_rows;  %valid row index
        
        range_cols = range_cols(v_col_index);   %range of cols. and rows satisfying euclidean distance metric  
        range_rows = range_rows(v_row_index);
        
        %current_vertex index
        p_vertex = n + (m - 1) * no_rows;
 %-----------------------------------------------------------------------------------------%       
        
        l1 = length(range_rows);
        l2 = length(range_cols);
        m1 = zeros(l1,l2);
        m2 = zeros(l1,l2);
        for i = 1:length(range_rows)
            for j = 1:length(range_cols)
                m1(i,j) = range_rows(i,1);
            end
        end
                   
        for i = 1:length(range_rows)
            for j = 1:length(range_cols)
                m2(i,j) = ((range_cols(1,j) -1) .*no_rows);
            end
        end
        n_vertex_temp = m1 + m2;    %dimensions l1 * l2
        n_vertex = zeros(l1*l2,1);
        temp = 1;
        for i = 1:l2
            for j = 1:l1
                n_vertex(temp,1) = n_vertex_temp(j,i);
                temp = temp + 1;        
            end
        end 
        
        %spatial location similarity
        X_J = zeros(length(n_vertex),1,2); 
        for k = 1:2
            for i = 1:length(n_vertex)
                X_J(i,1,k) = X(n_vertex(i,1),1, k);
            end
        end      
                
        
        X_I_temp = X(p_vertex, 1, :);
        X_I = zeros(length(n_vertex),1,2);  
      
        for i = 1:length(n_vertex)
            for k = 1:2
                X_I(i,1,k) = X_I_temp(1,1,k);
            end
        end
        diff_X = X_I - X_J;
        diff_X = sum(diff_X .* diff_X, 3); % squared euclid distance
        
        % |X(i) - X(j)| <= r 
        valid_index = (sqrt(diff_X) <= r);
        n_vertex = n_vertex(valid_index);
        diff_X = diff_X(valid_index);

        % feature vector disimilarity
        F_J = zeros(length(n_vertex),1,c); 
        for i = 1:length(n_vertex)
            for k = 1:c
                a = n_vertex(i,1);
                F_J(i,1,k) = F(a,1,k);
            end
        end
        F_J = uint8(F_J);
        
        FI_temp = F(p_vertex, 1, :);
        F_I = zeros(length(n_vertex),1,c);  
        for i = 1:length(n_vertex)
            for k = 1:c
                F_I(i,1,k) = FI_temp(1,1,k);
            end
        end
        F_I = uint8(F_I);        
        
        diff_F = F_I - F_J;
        diff_F = sum(diff_F .* diff_F, 3); 
        W(p_vertex, n_vertex) = exp(-diff_F / (sigma_I*sigma_I)) .* exp(-diff_X / (sigma_X*sigma_X)); % for squared distance
        
    end
end

% call to partition routine
node_index = (1:N)'; 
[node_index Ncut] = NcutPartition(node_index, W, threshold_Ncut, threshold_Area);


%  node_indexes to images

for i=1:length(node_index)
    Segment_I_temp = zeros(N, c);
    Segment_I_temp(node_index{i}, :) = V_node(node_index{i}, :);
    %Segment_I_temp1 = zeros(no_rows, no_cols, c);
    %size(Segment_I_temp)
    Segment_I_temp1{i} = (reshape(Segment_I_temp, no_rows, no_cols, c));
    Segment_I{i} = uint8(Segment_I_temp1{i});
    
end


for i=1:length(Segment_I)
figure; 
imshow(Segment_I{i});
imwrite(Segment_I{i}, sprintf('test%d.jpg', i));
fprintf('Ncut(%d) = %f\n', i, Ncut{i});
end
end

function [node_index Ncut] = NcutPartition(I, W, threshold_Ncut, threshold_Area)
N = length(W);
d = sum(W, 2);
D = sparse(N,N);
for i = 1:N
    D(i,i) = d(i);
end

[Y,lambda] = eigs(D-W, D, 2, 'sm'); % (D - W)Y = lambda * D * Y
eig_vector_2 = Y(:, 2);

split_point = median(eig_vector_2);  % starting point for fminsearch
split_point = fminsearch('NcutValue', split_point, split_point, eig_vector_2, W, D);

Partition_1 = find(eig_vector_2 > split_point);
Partition_2 = find(eig_vector_2 <= split_point);

Ncut_value = NcutValue(split_point, eig_vector_2, W, D);
if (length(Partition_1) < threshold_Area || length(Partition_2) < threshold_Area || Ncut_value > threshold_Ncut)
    node_index{1}   = I;
    Ncut{1} = Ncut_value; 
    return;
end

%recursive partition
[node_index_1 Ncut_1]  = NcutPartition(I(Partition_1), W(Partition_1, Partition_1), threshold_Ncut, threshold_Area);
[node_index_2 Ncut_2] = NcutPartition(I(Partition_2), W(Partition_2, Partition_2), threshold_Ncut, threshold_Area);

node_index   = cat(2, node_index_1, node_index_2);
Ncut = cat(2, Ncut_1, Ncut_2);
end

function value = NcutValue(split_point, eig_vector_2, W, D)

x = (eig_vector_2 > split_point);
x = (2 * x) - 1; %indicator rv's for Partitions 1 & 2
d = sum(W,2); 
k = sum(d(x>0))/sum(d);
b = k/(1 - k);
y = (1 + x) - b*(1 - x);

value = (y'*(D - W)*y)/(y'*D*y);

end


