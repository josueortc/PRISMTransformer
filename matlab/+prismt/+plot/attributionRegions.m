function attributionRegions(ax, I, opts)
%ATTRIBUTIONREGIONS Which channels carry class-dependent attribution, and which are significant.
%
%   prismt.plot.attributionRegions(ax, I, Modality="calcium")
%   Bars: summed |class difference| of the attribution over time; filled bars are
%   significant against label permutations (p < 0.05; * also after Benjamini-Hochberg).
%   These are the "PRISMt-selected" channels.
arguments
    ax
    I struct
    opts.Modality = 1
end
M = I.Mat;
m = opts.Modality;
if ~isnumeric(m), m = find(M.modality_names == string(m), 1); end
if isempty(m), m = 1; end
v = M.region_statistic(:, m);
p = M.region_p(:, m);
q = M.region_q(:, m);
prismt.plot.prepareAxes(ax);
s = prismt.plot.style();
n = numel(v);
sig = p < 0.05;
bar(ax, find(~sig), v(~sig), 'FaceColor', s.missing, 'EdgeColor', 'none', 'BarWidth', 0.8);
if any(sig), bar(ax, find(sig), v(sig), 'FaceColor', s.categorical(1, :), 'EdgeColor', 'none', 'BarWidth', 0.8); end
star = find(q < 0.05);
if ~isempty(star), text(ax, star, v(star), "*", 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom'); end
if n <= 60
    ax.XTick = 1:n; ax.XTickLabel = cellstr(M.channel_names); ax.XTickLabelRotation = 90;
end
xlim(ax, [0 n + 1]);
ylabel(ax, "Class-dependent attribution");
title(ax, sprintf("%s: %d of %d channels significant (p < 0.05)", M.modality_names(m), nnz(sig), n), ...
    'FontWeight', 'normal');
end
