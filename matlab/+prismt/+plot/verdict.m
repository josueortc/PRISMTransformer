function verdict(ax, R)
%VERDICT A run's result as answers: one line per question, with a colored mark.
%
%   prismt.plot.verdict(ax, prismt.loadResults(runDir))
%   Green: yes / fine.  Amber: look at this before concluding.  Red: a problem.  Gray: for
%   information. The answers come from the run's metrics.json (verdict), where each has
%   the numbers behind it; they are deliberately conservative (an interval that includes
%   "no effect" is never a yes).
arguments
    ax
    R struct
end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
axis(ax, 'off');
xlim(ax, [0 1]); ylim(ax, [0 1]);
if ~isfield(R.Metrics, 'verdict') || isempty(R.Metrics.verdict)
    text(ax, 0.02, 0.9, "This run was made before PRISMT wrote verdicts. Train it again to get one.", ...
        'Color', s.muted, 'VerticalAlignment', 'top');
    return
end
items = R.Metrics.verdict;
if iscell(items), items = [items{:}]; end
colors = struct('good', s.good, 'caution', s.warning, 'bad', s.critical, 'info', s.muted);
text(ax, 0.02, 0.985, "What the result says", 'FontSize', 13, 'FontWeight', 'bold', 'Color', s.ink, ...
    'VerticalAlignment', 'top');
n = numel(items);
step = min(0.19, 0.9 / n);
for k = 1:n
    it = items(k);
    y = 0.90 - (k - 1) * step;
    c = s.muted;
    if isfield(colors, char(it.status)), c = colors.(char(it.status)); end
    scatter(ax, 0.035, y - 0.012, 110, c, 'filled');
    text(ax, 0.07, y, string(it.question), 'FontSize', 11, 'FontWeight', 'bold', 'Color', s.ink, ...
        'VerticalAlignment', 'top', 'Interpreter', 'none');
    text(ax, 0.07, y - 0.038, string(it.answer), 'FontSize', 11, 'FontWeight', 'bold', 'Color', c * 0.85, ...
        'VerticalAlignment', 'top', 'Interpreter', 'none');
    text(ax, 0.07, y - 0.078, wrap(string(it.detail), 120), 'FontSize', 9, 'Color', s.muted, ...
        'VerticalAlignment', 'top', 'Interpreter', 'none');
end
end

function out = wrap(txt, width)
words = split(txt, " ")';
lines = strings(1, 0);
cur = "";
for w = words
    if strlength(cur) + strlength(w) + 1 > width && strlength(cur) > 0
        lines(end + 1) = cur; %#ok<AGROW>
        cur = w;
    elseif strlength(cur) == 0
        cur = w;
    else
        cur = cur + " " + w;
    end
end
lines(end + 1) = cur;
out = strjoin(lines(1:min(end, 3)), newline);
end
