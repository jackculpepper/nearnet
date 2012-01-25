
M = 5;
L = 3;
B = 2;

X = randn(L,B);
phi = randn(L,M);
a = randn(M,B);

alpha = 3;
beta = 4;

tic
checkgrad('objfun_phi', phi(:), 1e-4, X, a, alpha, beta)
toc

