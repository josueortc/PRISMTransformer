function name = recordingField(a)
%RECORDINGFIELD The field of one struct element that holds its continuous recording ("" if none).
%   Preferred names first (signal, data, dff, calcium, traces, recording); otherwise the
%   largest 2-D numeric field with more than one row and column.
name = "";
if ~isstruct(a) || ~isscalar(a), return; end
is2d = @(x) isnumeric(x) && ismatrix(x) && all(size(x) > 1);
for f = ["signal", "data", "dff", "calcium", "traces", "recording"]
    if isfield(a, f)
        v = a.(f);
        if iscell(v) && ~isempty(v), v = v{1}; end
        if is2d(v), name = f; return; end
    end
end
best = 0;
for f = string(fieldnames(a))'
    v = a.(f);
    if iscell(v) && ~isempty(v), v = v{1}; end
    if is2d(v) && numel(v) > best
        best = numel(v);
        name = f;
    end
end
end
