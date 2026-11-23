function motifBrain(axs, I, opts)
%MOTIFBRAIN One PRISMt motif's spatial profile on the channel map, one panel per signal.
%
%   prismt.plot.motifBrain(axs, I, Motif=1, X=ds.ChannelX, Y=ds.ChannelY)
%   prismt.plot.motifBrain(axs, I, Dataset=ds)        % the dataset's label image, atlas or positions
%   Each panel shows the motif's loading per channel times that signal's share, all on one
%   color scale, so a motif carried by one signal is pale in the other. Channels with
%   significant class-dependent attribution in that signal (p < 0.05) are outlined.
%   Without a layout the channels are drawn as bars (* = significant).
arguments
    axs
    I struct
    opts.Motif (1, 1) double {mustBePositive, mustBeInteger} = 1
    opts.Atlas (1, 1) string = ""
    opts.X double = []
    opts.Y double = []
    opts.Dataset = []
end
if ~iscell(axs), axs = num2cell(axs); end
M = I.Mat;
k = min(opts.Motif, size(M.motif_spatial, 2));
atlas = opts.Atlas; X = opts.X; Y = opts.Y; labels = [];
if ~isempty(opts.Dataset)
    ds = opts.Dataset;
    % the run may use a subset of the dataset's channels: place them by name
    [found, where] = ismember(M.channel_names, ds.ChannelNames);
    if all(found)
        whole = numel(M.channel_names) == numel(ds.ChannelNames) && all(where(:)' == 1:numel(where));
        if strlength(atlas) == 0 && strlength(ds.Atlas) && whole, atlas = ds.Atlas; end
        if ~isempty(ds.AtlasImage) && whole, labels = ds.AtlasImage; end
        if isempty(X) && ~isempty(ds.ChannelX), X = ds.ChannelX(where); Y = ds.ChannelY(where); end
    end
end
nM = numel(M.modality_names);
d = M.motif_spatial(:, k);
w = M.motif_modality(:, k);
lim = [0, max(d) * max(w)];
if lim(2) <= 0, lim = [0 1]; end
for m = 1:min(nM, numel(axs))
    v = d * w(m);
    if isfield(M, 'exists'), v(~logical(M.exists(:, m))) = NaN; end
    sig = false(numel(v), 1);
    if isfield(M, 'region_p'), sig = M.region_p(:, m) < 0.05; end
    ttl = sprintf("Motif %d — %s (%.0f%%)", k, M.modality_names(m), 100 * w(m));
    args = {'Title', ttl, 'Label', "Motif loading", 'Limits', lim, 'Signed', false, 'Highlight', sig};
    if ~isempty(prismt.internal.atlas(atlas, numel(v)))
        prismt.plot.channelMap(axs{m}, v, 'Atlas', atlas, args{:});
    elseif ~isempty(labels)
        prismt.plot.channelMap(axs{m}, v, 'Labels', labels, args{:});
    elseif ~isempty(X) && any(isfinite(X))
        prismt.plot.channelMap(axs{m}, v, 'X', X, 'Y', Y, 'Names', M.channel_names, args{:});
    else
        prismt.plot.channelMap(axs{m}, v, 'Names', M.channel_names, args{:});
    end
end
end
