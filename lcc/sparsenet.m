

for t = 1:num_trials

    P = randperm(size(X,2));
    Y = X(:,P(1:B));


    %% compute distance from every image to every basis element
    tic();

    if 0
        D = zeros(L,M,B);
        for m = 1:M
            for b = 1:B
                D(:,m,b) = Y(:,b) - phi(:,m);
            end
        end
    else
        D = bsxfun(@minus, reshape(Y, L, 1, B), reshape(phi, L, M, 1));
    end

    D2 = sum(D.^2);
    D2 = reshape(D2, M, B);

    time_dst = toc();



    tic();

    switch mintype
        case 'ldiv'
            a1 = zeros(M, B);
            for b = 1:B

                [val,idx] = sort(D2(:,b), 'ascend');
                %% select only the closest K basis functions
                idx = idx(1:K);

                if flag_gstats
                    if target_angle < 0.03
                        idx_key = sprintf('%u.', sort(idx));
                        if gstats.isKey(idx_key)
                            gstats(idx_key) = gstats(idx_key) + 1;
                        else
                            gstats(idx_key) = 0;
                        end
                    end
                end

                a1(idx,b) = (phi(:,idx)' * phi(:,idx) + beta * eye(K)) \ ...
                             phi(:,idx)' * Y(:,b);
            end
            %a = sparse(a1);
            a = a1;

        case 'lbfgsb'

            a1 = zeros(M, B);
            for b = 1:B
                a0 = zeros(M,1);

                [val,idx] = sort(D2(:,b), 'ascend');
                %% select only the closest K basis functions
                idx = idx(1:K);

                lb  = zeros(1,K); % lower bound
                ub  = zeros(1,K); % upper bound
                nb  = ones(1,K);  % bound type (lower only)
                nb  = zeros(1,K); % bound type (none)

                [a1(idx,b),fx,exitflag,userdata] = lbfgs(@objfun_a, a0(idx), ...
                    lb, ub, nb, opts, Y(:,b), phi(:,idx), D(:,idx,b), D2(idx,b), alpha, beta);

                fprintf(' %d / %d', b, B);
            end
            fprintf('\n');

            a = a1;
    end

    time_inf = toc();


    EI = phi*a;
    E = Y-EI;
    snr = 10 * log10 ( sum(Y(:).^2) / sum(E(:).^2) );


    tic()

    % update bases
    [obj0,g] = objfun_phi(phi(:), Y, a, alpha, beta, D, D2);

    dphi = reshape(g,L,M);

    phi1 = phi - eta*dphi;

    [obj1,dphi] = objfun_phi(phi1(:), Y, a, alpha, beta, D, D2);

    %% pursue a constant change in angle
    angle_phi = acos(phi1(:)' * phi(:) / sqrt(sum(phi1(:).^2)) / sqrt(sum(phi(:).^2)));
    if angle_phi < target_angle
        eta = eta*1.01;
    else
        eta = eta*0.99;
    end

    phi = phi1;

    time_lrn = toc();


    if (obj1 > obj0)
        fprintf('warning: objective function increased\n');
    end

    eta_log = eta_log(1:update-1);
    eta_log = [ eta_log ; eta ];


    %% display

    if (display_every == 1 || mod(update,display_every)==0)
        % Display the bfs
        array = render_network(phi, Mrows);
 
        sfigure(1); colormap(gray);
        imagesc(array, [-1 1]);
        axis image off;
 
        EI = phi*a(:,1);

        mx = max(abs([ EI(:) ; Y(:,1) ]));

        sfigure(4);
        subplot(1,2,1),imagesc(reshape(EI,Lsz,Lsz), [-mx mx]),title('EI');
            colormap(gray),axis image off;
        subplot(1,2,2),imagesc(reshape(Y(:,1),Lsz,Lsz),[-mx mx]),title('Y');
            colormap(gray),axis image off;

        sfigure(5);
        bar(a(:,1));
        axis tight;

        %fig_eta_log
        sfigure(6);
        plot(1:update, eta_log, 'r-');
        grid on;


        %% ranked distances of basis functions to image
        sfigure(7); bar(sqrt(val)); axis tight;

        if (save_every == 1 || mod(update,save_every)==0)
            array_frame = uint8(255*((array+1)/2)+1);
 
            imwrite(array_frame, ...
                sprintf('state/%s/phi_up=%06d.png',paramstr,update), 'png');
            eval(sprintf('save state/%s/phi.mat phi',paramstr));
            %eval(sprintf('save state/%s/phi_up=%06d.mat phi',paramstr,update));

            saveparamscmd = sprintf('save state/%s/params.mat', paramstr);
            saveparamscmd = sprintf('%s alpha', saveparamscmd);
            saveparamscmd = sprintf('%s beta', saveparamscmd);
            saveparamscmd = sprintf('%s tol_coef', saveparamscmd);
            saveparamscmd = sprintf('%s eta', saveparamscmd);
            saveparamscmd = sprintf('%s eta_log', saveparamscmd);
            saveparamscmd = sprintf('%s L', saveparamscmd);
            saveparamscmd = sprintf('%s M', saveparamscmd);
            saveparamscmd = sprintf('%s mintype', saveparamscmd);
            saveparamscmd = sprintf('%s update', saveparamscmd);
            eval(saveparamscmd);

        end
        drawnow;

    end

    % renormalize
    phi = phi*diag(1./sqrt(sum(phi.^2)));

    fprintf('update %d', update);
    fprintf(' %s', paramstr);
    fprintf(' dst %.3f', time_dst);
    fprintf(' inf %.3f', time_inf);
    fprintf(' lrn %.3f', time_lrn);
    fprintf(' l0 %.4f', full(mean(abs(sign(a(:))))) );
    fprintf(' ang %.3f', angle_phi);
    if flag_gstats
        fprintf(' grp %d', length(gstats.keys));
    end
    fprintf(' snr %.4f o0 %.8f o1 %.8f eta %.8f\n', snr, obj0, obj1, eta);

    update = update + 1;
end

%eval(sprintf('save state/%s/matlab_up=%06d.mat -v7.3', paramstr, update)); 

