% DeepReluNetworkPrior.m: Code supplement to 'Exact priors of finite neural networks'

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
    
    % Form a grid
    switch d
        case 2
            kGrid = (1:n(1,indN));
        case 3
            [k1,k2] = meshgrid((1:n(1,indN)),(1:n(2,indN)));
            kGrid = [k1(:)'; k2(:)'];
        case 4
            [k1,k2,k3] = meshgrid((1:n(1,indN)),(1:n(2,indN)),(1:n(3,indN)));
            kGrid = [k1(:)'; k2(:)'; k3(:)'];
        otherwise
            errror('d = %d is not supported.', d);
    end
    
    % Compute weights
    lnCk = sum(gammaln(n(:,indN)+1) - gammaln(kGrid+1) - gammaln(n(:,indN)-kGrid+1) - n(:,indN) * log(2),1);
    
    % Ignore weights less than threshold
    idx = lnCk > log(eps);
    kGrid = kGrid(:,idx);
    lnCk = lnCk(idx);
    
    pAcc = zeros(length(hVec),1);
    
    parfor indK = 1:size(kGrid,2)
        
        kVec = kGrid(:,indK);
        
        
        f = meijerG([], [], [0;(kVec-1)/2]', [], z);
        
        pLin = vpa(subs(f,z,prod(n(:,indN),1) .* hVec.^2 ./ (2^(2*d-1))).* sqrt(prod(n(:,indN),1)./(pi*(2^(2*d-1)))) ./ prod(gamma(kVec/2)));
        
        
        pAcc = pAcc + exp(lnCk(indK)) * pLin;
        
        fprintf('\tWidth %d of %d, term %d of %d\n', indN, size(n,2), indK, size(kGrid,2));
    end
    
    pVec(:,indN) = pAcc;
end

fprintf('Computed theoretical prior in %f seconds\n', toc);

% Compute infinite-width prediction
pInf = exp(-hVec.^2 ./ 2) ./ sqrt(2*pi);

%% Compute numerical estimate

tic;

x = nan(nRep,size(n,2));
for indN = 1:size(n,2)
    parfor indR = 1:nRep
        
        switch d
            case 2
                x(indR,indN) = randn(1, n(indN)) * max(0,randn(n(indN), 1)) * sqrt(2^(d-1) / n(indN));
            case 3
                x(indR,indN) = randn(1, n(2,indN)) * max(0,randn(n(2,indN),n(1,indN)) * max(0,randn(n(1,indN), 1))) * sqrt(2^(d-1) / prod(n(:,indN),1));
            case 4
                x(indR,indN) = randn(1, n(3,indN)) * max(0,randn(n(3,indN),n(2,indN)) * max(0,randn(n(2,indN),n(1,indN)) * max(0,randn(n(1,indN), 1)))) * sqrt(2^(d-1) / prod(n(:,indN),1));
            otherwise
                errror('d = %d is not supported.', d);
        end
    end
end

fprintf('Computed numerical prior in %f seconds\n', toc);


%% Save data to file

if onCluster
    
    fpath = fullfile('~/prior_results',sprintf('deep_relu_network_prior_results_depth_%d_%s.mat', ...
        d, datestr(datetime('now'), 'yyyymmdd_HH_MM_SS')));
    save(fpath, 'd', 'n', 'nRep', 'hVec','pVec','pInf','x','-v7.3');
    
    fprintf('Saved data to %s\n', fpath);
    
end

