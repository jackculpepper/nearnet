
        [sucess,msg,msgid] = mkdir(sprintf('cache', paramstr));

        filename = sprintf('cache/X_%dx%d.mat', L, C);
        if exist( filename, 'file')
            cache = load(filename);
            X = cache.X;
        else

            load ../data/IMAGES.mat
            [Wsz,Wsz,Ksz] = size(IMAGES);

            X = zeros(L,C);

            % extract subimages at random
            fprintf('selecting image patches ..\n');
            for b = 1:C
                while std(X(:,b)) < 0.1
                    i = ceil(Ksz*rand);
                    r = buff + ceil((Wsz-Lsz-2*buff)*rand);
                    c = buff + ceil((Wsz-Lsz-2*buff)*rand);

                    X(:,b) = reshape(IMAGES(r:r+Lsz-1,c:c+Lsz-1,i), L, 1);
                end

                X(:,b) = X(:,b) - mean(X(:,b));
                X(:,b) = X(:,b) / std(X(:,b));

                fprintf('\r%d / %d', b, C);
            end
            fprintf('\n');

            clear IMAGES
               
            %% save to cache
            [sucess,msg,msgid] = mkdir('cache');
            save(filename, 'X', '-v7.3');

        end


