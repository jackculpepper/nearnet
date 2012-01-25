
M = 5;
L = 3;
B = 2;

Y = randn(L,B);
phi = randn(L,M);
a = randn(M,B);

alpha = 3;
beta = 4;


%% compute distance from every image to every basis element
D = zeros(L,M,B);
for m = 1:M
    for b = 1:B
        D(:,m,b) = Y(:,b) - phi(:,m);
    end
end
D2 = sum(D.^2);
D2 = reshape(D2, M, B);



tic
checkgrad('objfun_a', a(:), 1e-4, Y, phi, D, D2, alpha, beta)
toc

