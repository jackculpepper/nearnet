clear

M = 1600; Mrows = 20;
M = 512; Mrows = 16;
M = 1024; Mrows = 16;
M = 256; Mrows = 16;
M = 2048; Mrows = 32;
M = 4000; Mrows = 40;
L = 100;
L = 400;

Lsz = sqrt(L);

tol_coef = 0.01;

save_every = 10;
save_every = 100;

alpha = 1.0;
beta = 0.01;
beta = 0.01;

buff = 4;

mintype = 'lbfgsb';
mintype = 'ldiv';

switch mintype
    case 'lbfgsb'
        opts = lbfgs_options('iprint', -1, 'maxits', 20, ...
                             'factr', 1e-1, ...
                             'cb', @cb);
end


test_every = 20;

paramstr = sprintf('L=%03d_M=%03d_%s',L,M,datestr(now,30));
[sucess,msg,msgid] = mkdir(sprintf('state/%s', paramstr));

C = 100000;
C = 1000000;

reinit

eta_log = [];
objtest_log = [];

eta = 0.1;
eta = 0.01;
eta = 0.005;

target_angle = 0.1;
target_angle = 0.05;
target_angle = 0.04;


display_every = 10;
display_every = 100;

flag_gstats = 0;

if flag_gstats
    gstats = containers.Map();
end


%% initialization phase

num_trials = 1000;
B = 10;
K = 200;
sparsenet

num_trials = 1000;
B = 10;
K = 100;
sparsenet


%% reduce K to where we want it, and begin learning/annealing

num_trials = 10000;
B = 10;
K = 20;

for target_angle = 0.04:-0.005:0.005 ; sparsenet ; end


