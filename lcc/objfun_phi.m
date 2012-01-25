function [f,g] = objfun_phi(x0, X, a, alpha, beta, D, D2);

[L B] = size(X);
M = size(a,1);

phi = reshape(x0, L, M);

R = X - phi*a;


f = 0.5*alpha*sum(R(:).^2);
f = f + beta*sum( D2(:) .* abs(a(:)) );

if 0
    g = -alpha * R*a';

    for l = 1:L
        for m = 1:M
            for b = 1:B
                g(l,m) = g(l,m) - 2 * beta * D(l,m,b) * abs(a(m,b));
            end
        end
    end
else

    if 1
        Da = bsxfun(@times, D, shiftdim(abs(a), -1));
        g = -alpha * R*a' - 2*beta*sum(Da, 3);

    else
        g = -alpha * R*a';

        for l = 1:L
            Da = squeeze(D(l,:,:)).*abs(a);
            g(l,:) = g(l,:) - 2*beta*sum(Da,2)';
        end
    end
end

g = g(:);

