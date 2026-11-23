function p = fromContinuous(S, vars, opts)
%FROMCONTINUOUS Read continuous recordings and cut them into windows.
%   Each variable in vars is a struct array with one element per subject, holding a
%   channels x time recording; each variable is one group of subjects (for example two
%   genotypes). Recordings are cut into windows of opts.Window samples (step opts.Stride,
%   default = Window). The trial column "group" says which variable a subject came from:
%   the part of the variable names that differs (rec_wt_struct, rec_mut_struct -> wt, mut).
p = prismt.import.emptyParts();
stride = opts.Stride;
if isempty(stride), stride = opts.Window; end
groups = groupLabels(vars);
k = 0;
for g = 1:numel(vars)
    A = S.(vars(g));
    if istable(A), A = table2struct(A); end
    for i = 1:numel(A)
        a = A(i);
        fld = prismt.import.recordingField(a);
        if strlength(fld) == 0, continue; end
        data = a.(fld);
        if iscell(data), data = data{1}; end
        if isempty(data), continue; end
        k = k + 1;
        % Continuous recordings are channels x time; windowing wants time x channels.
        W = prismt.import.windowContinuous(double(data'), opts.Window, stride);
        p.signals.dff{k, 1} = W;
        subject = sprintf('%s_%d', groups(g), i);
        for id = ["mouse", "mouse_id", "animal_id", "subject", "subject_id", "id"]
            if isfield(a, id), v = a.(id); if iscell(v), v = v{1}; end, subject = char(string(v)); break; end
        end
        p.sessionCols.mouse(k, 1) = string(subject);
        p.sessionCols.group(k, 1) = groups(g);
    end
end
p.sessionNames = compose("%s_rec", p.sessionCols.mouse);
p.notes(end + 1) = sprintf("Continuous recordings cut into windows of %d samples every %d samples.", opts.Window, stride);
if numel(vars) > 1
    p.notes(end + 1) = "The column 'group' says which variable each subject came from (" + strjoin(groups, ", ") + ").";
end
end

function g = groupLabels(vars)
% The part of the variable names that differs: the words (separated by "_") that all
% names share at the start and at the end are removed (rec_wt_struct, rec_mut_struct -> wt, mut).
g = string(vars(:))';
if numel(g) < 2, return; end
parts = arrayfun(@(s) split(s, "_")', g, 'UniformOutput', false);
n = min(cellfun(@numel, parts));
pre = 0;
while pre < n - 1 && all(cellfun(@(p) p(pre + 1) == parts{1}(pre + 1), parts)), pre = pre + 1; end
suf = 0;
while suf < n - pre - 1 && all(cellfun(@(p) p(end - suf) == parts{1}(end - suf), parts)), suf = suf + 1; end
out = cellfun(@(p) strjoin(p(pre + 1:end - suf), "_"), parts);
if all(strlength(out) > 0) && numel(unique(out)) == numel(out), g = out; end
end
