function array = render_network_color(A, rows, num_colors)

[L M] = size(A);

sz = sqrt(L/num_colors);

buf = 1;

m = rows;
n = M/rows;

array = zeros(buf+m*(sz+buf), buf+n*(sz+buf), num_colors);

k = 1;

for i = 1:m
    for j = 1:n
        B = A(:,k);
        B = B - min(B(:));
        B = B / max(B(:));

        array(buf+(i-1)*(sz+buf)+[1:sz],buf+(j-1)*(sz+buf)+[1:sz],:) = ...
            reshape(B,sz,sz,num_colors);

        k = k+1;
    end
end


