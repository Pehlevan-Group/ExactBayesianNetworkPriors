% DeepLinearNetworkPrior.m: Code supplement to 'Exact priors of finite neural networks'


% Depth
d = 2;

% Widths of each hidden layer (should be a (d-1) x (number of widths) array)
n = [
    1,2,5,10,100;
    ];

% Number of samples for numerical estimate
nRep = 5e7;

% Points at which to estimate the prior
hVec = (0.01:0.01:5)';

% Use slurm environment variables as a proxy for whether the script is
% running on the cluster
onCluster = ~isempty(getenv('SLURM_CPUS_PER_TASK'));

%% Set up parallel pool

poolObj = gcp('nocreate');
if isempty(poolObj)
    if onCluster
        poolObj = parpool('local', str2double(getenv('SLURM_CPUS_PER_TASK')));
    else
        poolObj = parpool('local');
    end
end
rng('shuffle');


%% Compute analytical prediction

tic;

pVec = nan(length(hVec), size(n,2));

syms z;

for indN = 1:size(n,2)
    
    f = meijerG([], [], [0;(n(:,indN)-1)/2]', [], z);
    pVec(:,indN) = vpa(subs(f,z,prod(n(:,indN),1) .* hVec.^2 ./ (2^d)).* sqrt(prod(n(:,indN),1)./(pi*(2.^d))) ./ prod(gamma(n(:,indN)/2),1));
    
end

toc;

% Compute infinite-width prediction
pInf = exp(-hVec.^2 ./ 2) ./ sqrt(2*pi);

%% Compute numerical estimate

tic;

x = nan(nRep,size(n,2));
for indN = 1:size(n,2)
    parfor indR = 1:nRep
        
        switch d
            case 2
                x(indR,indN) = randn(1, n(indN)) * randn(n(indN), 1) / sqrt(n(indN));
            case 3
                x(indR,indN) = randn(1, n(2,indN)) * randn(n(2,indN),n(1,indN)) * randn(n(1,indN), 1) / sqrt(prod(n(:,indN),1));
            case 4
                x(indR,indN) = randn(1, n(3,indN)) * randn(n(3,indN),n(2,indN)) * randn(n(2,indN),n(1,indN)) * randn(n(1,indN), 1) / sqrt(prod(n(:,indN),1));
            otherwise
                errror('d = %d is not supported.', d);
        end
    end
end

toc;


%% Save data to file

if onCluster
    
    fpath = fullfile('~/prior_results',sprintf('deep_linear_network_prior_results_depth_%d_%s.mat', ...
       d, datestr(datetime('now'), 'yyyymmdd_HH_MM_SS')));
    save(fpath, 'd', 'n', 'nRep', 'hVec','pVec','pInf','x','-v7.3');
    
    fprintf('Saved data to %s\n', fpath);
    
end


