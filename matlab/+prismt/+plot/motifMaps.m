function img = motifMaps(ax, I, opts)
%MOTIFMAPS One PRISMt motif as a channel x time map for each signal (side by side).
%
%   prismt.plot.motifMaps(ax, prismt.loadInterpretation(runDir))          % motif 1
%   prismt.plot.motifMaps(ax, I, Motif=2)
%   A motif is spatial profile x time course x share per signal; the map shows that
%   product, so a motif carried by one signal is empty in the other.
arguments
    ax
    I struct
    opts.Motif (1, 1) double {mustBePositive, mustBeInteger} = 1
end
M = I.Mat;
k = min(opts.Motif, size(M.motifs, 1));
A = squeeze(M.motifs(k, :, :, :));
if size(M.motifs, 4) == 1, A = reshape(A, size(M.motifs, 2), size(M.motifs, 3), 1); end
[img, xt] = rtmImage(A);
prismt.plot.prepareAxes(ax);
s = prismt.plot.style();
imagesc(ax, img, 'AlphaData', ~isnan(img));
colormap(ax, s.sequential);
set(ax, 'YDir', 'normal');
nC = size(A, 1);
ax.XTick = xt;
ax.XTickLabel = cellstr(M.modality_names);
if nC <= 40
    ax.YTick = 1:nC; ax.YTickLabel = cellstr(M.channel_names);
end
ylabel(ax, "Channel");
xlabel(ax, "Time within each signal");
w = M.motif_modality(:, k);
call = "";
if ~isempty(I.Summary) && isfield(I.Summary, 'motifs')
    mm = I.Summary.motifs;
    if iscell(mm), mm = [mm{:}]; end
    c = mm(k).modality;
    call = string(c.point);
    if isfield(c, 'interval') && ~isempty(c.interval), call = string(c.interval); end
end
share = strjoin(compose("%s %.0f%%", M.modality_names, 100 * w), ", ");
use = "";
if exist('mm', 'var') && isfield(mm(k), 'test') && isstruct(mm(k).test) && isfield(mm(k).test, 'used') && ~isempty(mm(k).test.used)
    if mm(k).test.used, use = "; the model uses it"; else, use = "; no evidence the model uses it"; end
end
title(ax, sprintf("Motif %d of %d — %s (%s%s)", k, size(M.motifs, 1), share, describe(call, M.modality_names), use), ...
    'FontWeight', 'normal');
cb = colorbar(ax); cb.Label.String = "Motif loading";
axis(ax, 'tight');
end

function [img, xt] = rtmImage(A)
[C, T, Mn] = size(A);
img = nan(C, T * Mn + (Mn - 1));
xt = zeros(1, Mn);
for m = 1:Mn
    cols = (m - 1) * (T + 1) + (1:T);
    img(:, cols) = A(:, :, m);
    xt(m) = mean(cols);
end
end

function s = describe(call, names)
s = call;
if startsWith(call, "unique:")
    s = names(str2double(extractAfter(call, "unique:")) + 1) + " only";
end
end
