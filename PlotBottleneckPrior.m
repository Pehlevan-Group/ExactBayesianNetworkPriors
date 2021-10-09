% PlotBottleneckPrior.m: Code supplement to 'Exact priors of finite neural networks'

% Depth
d = 4;

% Width of the non-bottleneck hidden layers
n0 = 100;

% Width of the bottleneck layer
k = [1,2,5,10,50,100];

% Points at which to estimate the prior
hVec = (0.01:0.01:5)';

%%

% Color order
corder = cbrewer('qual','Dark2',6);
corder = corder([1,2,3,4,6,5],:);

%% Compute analytical prediction

tAll = tic;

pVec = nan(length(hVec), size(k,2));

syms z;

for indK = 1:size(k,2)
    tic; 
    n = [n0;k(indK);n0];
    f = meijerG([], [], [0;(n-1)/2]', [], z);
    pVec(:,indK) = vpa(subs(f,z,prod(n) .* hVec.^2 ./ (2^d)).* sqrt(prod(n)./(pi*(2.^d))) ./ prod(gamma(n/2)));
    fprintf('\tn%d of %d: %f seconds\n', indK, size(k,2),toc);
end

toc(tAll);

% Compute infinite-width prediction
pInf = exp(-hVec.^2 ./ 2) ./ sqrt(2*pi);

%%

MakeFigure;
hold on;
set(gca, 'colororder', corder);
plot(hVec, pVec, 'linewidth', 2);
plot(hVec, pInf, '-k','linewidth', 2);
set(gca, 'ColorOrderIndex',1);

legend([strip(cellstr(num2str(k'))); '\infty']);

xlim([0,max(hVec)]);
set(gca, 'yscale','log');
ylim([1e-5,1e1]);

xlabel('h');
ylabel('p(h)');
title(sprintf('n_0 = %d, d=%d', n0, d));
axis('square');
set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off');

