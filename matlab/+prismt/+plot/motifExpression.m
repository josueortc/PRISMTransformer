function motifExpression(ax, I, opts)
%MOTIFEXPRESSION How strongly each trial expresses a motif, by class (or subject, session).
%
%   prismt.plot.motifExpression(ax, I)                    % motif 1, by class
%   prismt.plot.motifExpression(ax, I, Motif=2, By="subject")
%   Available when the interpretation kept the trial axis (Mode "trials", the default for
%   recordings).
arguments
    ax
    I struct
    opts.Motif (1, 1) double {mustBePositive, mustBeInteger} = 1
    opts.By (1, 1) string {mustBeMember(opts.By, ["class", "subject", "session"])} = "class"
end
M = I.Mat;
prismt.plot.prepareAxes(ax);
if ~isfield(M, 'motif_expression')
    title(ax, "No per-trial expression: this interpretation used the class difference", 'FontWeight', 'normal');
    axis(ax, 'off');
    return
end
k = min(opts.Motif, size(M.motif_expression, 2));
v = M.motif_expression(:, k);
switch opts.By
    case "class", g = M.classes(M.trial_class);
    case "subject", g = M.trial_subject;
    otherwise, g = M.trial_session;
end
g = categorical(g(:));
s = prismt.plot.style();
cats = categories(g);
for j = 1:numel(cats)
    sel = g == cats{j};
    x = j + 0.25 * (rand(nnz(sel), 1) - 0.5);
    scatter(ax, x, v(sel), 10, s.categorical(mod(j - 1, 8) + 1, :), 'filled', 'MarkerFaceAlpha', 0.35);
    plot(ax, j + [-0.3 0.3], median(v(sel)) * [1 1], '-', 'Color', s.ink, 'LineWidth', 2);
end
ax.XTick = 1:numel(cats); ax.XTickLabel = cats;
xlim(ax, [0.4 numel(cats) + 0.6]);
ylabel(ax, sprintf("Expression of motif %d", k));
title(ax, sprintf("Motif %d per trial (line: median)", k), 'FontWeight', 'normal');
end
