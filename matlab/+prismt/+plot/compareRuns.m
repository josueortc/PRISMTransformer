function compareRuns(ax, C)
%COMPARERUNS The scores of several runs side by side, with their intervals.
%
%   C = prismt.compareRuns([runA, runB]);  prismt.plot.compareRuns(ax, C)
%   Bars: each run's score on its held-out trials; error bars: 95% bootstrap interval
%   (resampling subjects); dotted line: chance. C.summary_lines gives the paired
%   differences to the first run.
arguments
    ax
    C struct
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
runs = C.runs;
if iscell(runs), runs = [runs{:}]; end
n = numel(runs);
vals = nan(n, 1); lo = nan(n, 1); hi = nan(n, 1); names = strings(n, 1);
for k = 1:n
    if ~isempty(runs(k).value), vals(k) = runs(k).value; end
    if isfield(runs(k), 'ci') && numel(runs(k).ci) == 2, lo(k) = runs(k).ci(1); hi(k) = runs(k).ci(2); end
    names(k) = string(runs(k).name);
end
b = barh(ax, vals, 0.6, 'FaceColor', 'flat', 'EdgeColor', 'none');
b.CData = s.categorical(mod((1:n) - 1, 8) + 1, :);
ok = ~isnan(lo);
if any(ok)
    errorbar(ax, vals(ok), find(ok), vals(ok) - lo(ok), hi(ok) - vals(ok), 'horizontal', 'Color', s.ink, ...
        'LineStyle', 'none', 'LineWidth', 1);
end
for k = 1:n
    if ~isnan(vals(k))
        x = vals(k); if ~isnan(hi(k)), x = hi(k); end
        text(ax, x + 0.015, k, sprintf('%.2f', vals(k)), 'Color', s.ink, 'VerticalAlignment', 'middle');
    end
end
yticks(ax, 1:n); yticklabels(ax, names); ax.YDir = 'reverse'; ax.TickLabelInterpreter = 'none';
if isfield(runs(1), 'chance') && ~isempty(runs(1).chance)
    xline(ax, runs(1).chance, ':', 'Color', s.muted);
end
if string(runs(1).score) == "balanced_accuracy", xlim(ax, [0 1.1]); end
xlabel(ax, replace(string(runs(1).score), "_", " ") + " on held-out trials");
title(ax, "Runs compared (bars: score; lines: 95% interval)", 'FontWeight', 'normal');
grid(ax, 'on'); ax.YGrid = 'off';
end
