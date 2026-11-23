function modalityAllocation(ax, I)
%MODALITYALLOCATION Each motif's share per signal, with its bootstrap interval.
%
%   prismt.plot.modalityAllocation(ax, prismt.loadInterpretation(runDir))
%   A motif is called specific to a signal when that signal's share is above 65% (and,
%   with bootstrap intervals, when the whole interval is), shared otherwise.
arguments
    ax
    I struct
end
M = I.Mat;
W = M.motif_modality';                 % K x signals
K = size(W, 1);
prismt.plot.prepareAxes(ax);
s = prismt.plot.style();
b = bar(ax, 1:K, W, 'stacked', 'EdgeColor', 'none');
for m = 1:numel(b), b(m).FaceColor = s.categorical(m, :); end
if isfield(M, 'modality_lo') && size(W, 2) >= 1 && ~isempty(M.modality_lo)
    lo = M.modality_lo(:, 1); hi = M.modality_hi(:, 1);
    errorbar(ax, 1:K, W(:, 1), W(:, 1) - lo, hi - W(:, 1), 'k', 'LineStyle', 'none', 'LineWidth', 1);
end
if size(W, 2) == 2
    plot(ax, [0.4 K + 0.6], [0.65 0.65], ':', 'Color', s.muted);
    plot(ax, [0.4 K + 0.6], [0.35 0.35], ':', 'Color', s.muted);
end
ylim(ax, [0 1]); xlim(ax, [0.4 K + 0.6]);
ax.XTick = 1:K; ax.XTickLabel = compose("Motif %d", 1:K);
ylabel(ax, "Share of the motif per signal");
legend(ax, b, cellstr(M.modality_names), 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off');
title(ax, "Which signal carries each motif (error bar: 95% interval of the first signal's share)", 'FontWeight', 'normal');
end
