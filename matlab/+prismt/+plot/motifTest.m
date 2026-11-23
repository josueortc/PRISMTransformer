function motifTest(ax, I)
%MOTIFTEST Does the model use each motif? The loss of confidence when its footprint is removed.
%
%   prismt.plot.motifTest(ax, prismt.loadInterpretation(runDir))
%   For each motif, the values where it is strongest (above 30% of its peak) are removed
%   from every held-out trial and the drop in the model's confidence in the right class is
%   measured (colored bar), next to the drop for as many randomly chosen values (gray bar,
%   with +-2 sd). A motif counts as used when its drop is reliably larger than the random
%   one and at least 2% of the model's evidence. A motif shows what the model attends to;
%   this test shows whether the prediction depends on it.
arguments
    ax
    I struct
end
M = I.Mat;
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
if ~isfield(M, 'motif_test_cost')
    title(ax, "This interpretation has no motif tests: press Interpret (PRISMt) again", 'FontWeight', 'normal');
    axis(ax, 'off');
    return
end
cost = M.motif_test_cost(:); ctrl = M.motif_test_control(:); sd = M.motif_test_control_sd(:);
K = numel(cost);
used = false(K, 1);
if ~isempty(I.Summary) && isfield(I.Summary, 'motifs')
    mm = I.Summary.motifs;
    if iscell(mm), mm = [mm{:}]; end
    for k = 1:K
        if isfield(mm(k), 'test') && isfield(mm(k).test, 'used') && ~isempty(mm(k).test.used)
            used(k) = logical(mm(k).test.used);
        end
    end
end
b = bar(ax, (1:K) - 0.2, cost, 0.36, 'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = repmat(s.missing, K, 1);
b.CData(used, :) = repmat(s.categorical(1, :), nnz(used), 1);
bar(ax, (1:K) + 0.2, ctrl, 0.36, 'FaceColor', s.muted * 0.4 + 0.6, 'EdgeColor', 'none');
errorbar(ax, (1:K) + 0.2, ctrl, 2 * sd, 'Color', s.ink, 'LineStyle', 'none');
for k = 1:K
    lab = "not used";
    if used(k), lab = "used"; end
    text(ax, k - 0.2, max(cost(k), 0), lab, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'Color', s.ink, 'FontSize', 9);
end
ax.XTick = 1:K; ax.XTickLabel = compose("Motif %d", 1:K);
xlim(ax, [0.4 K + 0.6]);
ylabel(ax, "Drop in the model's confidence when removed");
legend(ax, ["the motif's footprint", "as many random values (+-2 sd)"], 'Location', 'northoutside', ...
    'Orientation', 'horizontal', 'Box', 'off');
title(ax, "Does the model use each motif?", 'FontWeight', 'normal');
end
