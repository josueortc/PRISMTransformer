function channelMap(ax, values, opts)
%CHANNELMAP Draw one value per channel on a map, or as bars when there is no layout.
%
%   prismt.plot.channelMap(ax, v, Labels=ds.AtlasImage)          % label image (pixel k = channel k)
%   prismt.plot.channelMap(ax, v, X=ds.ChannelX, Y=ds.ChannelY)  % positions -> tiles
%   prismt.plot.channelMap(ax, v, Atlas=name)                    % a named atlas (atlas add-on)
%   prismt.plot.channelMap(ax, v, Names=ds.ChannelNames)         % no layout -> bars
%   prismt.plot.channelMap(ax, v, ..., Highlight=mask)           % outline these channels
%   Signed values use the diverging map with symmetric limits; others the blue ramp.
arguments
    ax
    values (:, 1) double
    opts.Atlas (1, 1) string = ""
    opts.Labels = []
    opts.X double = []
    opts.Y double = []
    opts.Names (:, 1) string = strings(0, 1)
    opts.Title (1, 1) string = ""
    opts.Label (1, 1) string = ""
    opts.Limits double = []
    opts.Signed = []
    opts.Highlight logical = false(0, 1)
end
hl = false(numel(values), 1);
if numel(opts.Highlight) == numel(values), hl = opts.Highlight(:); end
s = prismt.plot.style();
prismt.plot.prepareAxes(ax);
signed = opts.Signed;
if isempty(signed), signed = any(values < 0) && any(values > 0); end
lim = opts.Limits;
if isempty(lim)
    if signed, lim = [-1 1] * max(abs(values), [], 'omitnan'); else, lim = [min(values, [], 'omitnan') max(values, [], 'omitnan')]; end
end
cmap = s.sequential; if signed, cmap = s.diverging; end
labels = [];
A = prismt.internal.atlas(opts.Atlas, numel(values));
if ~isempty(A) && isfield(A, 'Labels') && ~isempty(A.Labels)
    labels = double(A.Labels);
elseif ~isempty(opts.Labels) && max(double(opts.Labels(:))) <= numel(values)
    labels = double(opts.Labels);
elseif ~isempty(A) && isempty(opts.X)
    opts.X = A.X; opts.Y = A.Y;            % an atlas without a label image: its centroids
end
if ~isempty(labels)
    img = nan(size(labels));
    inside = labels > 0;
    img(inside) = values(labels(inside));
    imagesc(ax, img, 'AlphaData', ~isnan(img));
    if any(hl)
        marked = false(size(labels));
        marked(inside) = hl(labels(inside));
        contour(ax, double(marked), [0.5 0.5], 'Color', s.ink, 'LineWidth', 1.2);
    end
    axis(ax, 'image', 'off'); ax.YDir = 'reverse';
elseif ~isempty(opts.X) && ~isempty(opts.Y) && any(isfinite(opts.X))
    ok = isfinite(opts.X) & isfinite(opts.Y);
    scatter(ax, opts.X(ok), opts.Y(ok), 900, values(ok), 's', 'filled', 'MarkerEdgeColor', s.surface, 'LineWidth', 2);
    if any(hl & ok)
        scatter(ax, opts.X(hl & ok), opts.Y(hl & ok), 900, 's', 'MarkerEdgeColor', s.ink, 'LineWidth', 1.5);
    end
    if ~isempty(opts.Names)
        text(ax, opts.X(ok), opts.Y(ok), opts.Names(ok), 'HorizontalAlignment', 'center', 'FontSize', 7, 'Color', s.ink);
    end
    ax.YDir = 'reverse'; axis(ax, 'equal', 'off');
    xlim(ax, [min(opts.X(ok)) - 0.7, max(opts.X(ok)) + 0.7]); ylim(ax, [min(opts.Y(ok)) - 0.7, max(opts.Y(ok)) + 0.7]);
else
    b = bar(ax, values, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.8);
    idx = round(rescale(values, 1, 256, 'InputMin', lim(1), 'InputMax', lim(2)));
    idx(~isfinite(idx)) = 1;
    b.CData = cmap(idx, :);
    if any(hl)
        text(ax, find(hl), max(values(hl), 0), "*", 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'Color', s.ink, 'FontSize', 12);
    end
    if ~isempty(opts.Names) && numel(values) <= 60
        xticks(ax, 1:numel(values)); xticklabels(ax, opts.Names); xtickangle(ax, 60);
    end
    ylabel(ax, opts.Label); grid(ax, 'on'); ax.XGrid = 'off';
end
colormap(ax, cmap);
prismt.plot.setLimits(ax, lim);
cb = colorbar(ax); cb.Color = s.muted; cb.Label.String = opts.Label;
title(ax, opts.Title);
end
