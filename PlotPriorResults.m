% PlotPriorResults.m: Code supplement to 'Exact priors of finite neural networks'

% Color order
corder = cbrewer('qual','Dark2',size(n,2));

%%

% Bin edges for empirical histograms
binEdges = (-6:0.025:6)';

MakeFigure;
hold on;
set(gca, 'colororder', corder);
plot(hVec, pVec, 'linewidth', 2);
plot(hVec, pInf, '-k','linewidth', 2);

% As we only plot one side of the distribution, we can use x and -x separately
for indN = 1:size(n,2)
    histogram([x(:,indN);-x(:,indN)], binEdges, 'displaystyle','stairs','normalization','pdf','linewidth',1, 'EdgeColor',corder(indN,:));
end

legend([strip(cellstr(num2str(n(1,:)'))); '\infty']);

xlim([0,max(hVec)]);
set(gca, 'yscale','log');
ylim([1e-5,1e1]);

xlabel('h');
ylabel('p(h)');
title(num2str(d,'linear network, d=%d'));
axis('square');
set(gca, 'FontSize', 16, 'LineWidth', 2, 'Box','off');

