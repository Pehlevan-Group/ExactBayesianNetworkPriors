% PlotEdgeworthApproximation.m: Code supplement to 'Exact priors of finite neural networks'


% Depth
d = 2;

% Widths of each hidden layer (should be a (d-1) x (number of widths) array)
n = [
    10,50,100;
    ];

% Points at which to estimate the prior
hVec = (0.01:0.01:5)';

%%

% Color order
corder = cbrewer('qual','Dark2',6);
corder = corder([4,6,5],:);

%% Compute analytical prediction

tAll = tic;

pVec = nan(length(hVec), size(n,2));

syms z;

for indN = 1:size(n,2)
   tic; 
    f = meijerG([], [], [0;(n(:,indN)-1)/2]', [], z);
    pVec(:,indN) = vpa(subs(f,z,prod(n(:,indN),1) .* hVec.^2 ./ (2^d)).* sqrt(prod(n(:,indN),1)./(pi*(2.^d))) ./ prod(gamma(n(:,indN)/2),1));
    fprintf('\tn%d of %d: %f seconds\n', indN, size(n,2),toc);
end

toc(tAll);

% Compute Edgeworth expansion
pEdgeworth = exp(-hVec.^2/2) .* (1 + sum(1./n,1) .* (hVec.^4 - 6 .* hVec.^2 + 3) / 4 ) / sqrt(2*pi);


% Compute infinite-width prediction
pInf = exp(-hVec.^2 ./ 2) ./ sqrt(2*pi);

%%

MakeFigure;
hold on;
set(gca, 'colororder', corder);
plot(hVec, pVec, 'linewidth', 2);
plot(hVec, pInf, '-k','linewidth', 2);
set(gca, 'ColorOrderIndex',1);
plot(hVec, pEdgeworth, '--', 'linewidth', 2);

legend([strip(cellstr(num2str(n(1,:)'))); '\infty']);

xlim([0,max(hVec)]);
set(gca, 'yscale','log');
ylim([1e-5,1e1]);

xlabel('h');
ylabel('p(h)');
title(num2str(d,'d=%d'));
axis('square');
set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off');

