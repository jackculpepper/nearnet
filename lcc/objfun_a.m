function [f,g] = objfun_a(x0, X, phi, D, D2, alpha, beta);

[L B] = size(X);
M = size(phi,2);

a = reshape(x0, M, B);

R = X - phi*a;

f = 0.5*alpha*sum(R(:).^2) + beta*sum( D2(:) .* abs(a(:)) );
g = -alpha*phi'*R + beta*D2.*sign(a);

g = g(:);

