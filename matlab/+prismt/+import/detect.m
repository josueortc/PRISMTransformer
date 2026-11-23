function [kind, var] = detect(S)
%DETECT Which known layout a loaded .mat struct has, and the variable(s) that hold it.
%   prismt        a PRISMT dataset file
%   table         a table with one row per session
%   structs       a struct with one numbered sub-struct per session (name_001, name_002, ...)
%   continuous    struct arrays (one element per subject) holding continuous recordings;
%                 var lists the variables, one per group of subjects
%   numbered      numbered variables (dff_001, stim_001, ...)
f = string(fieldnames(S));
var = strings(1, 0);
if ismember("prismt_format", f)
    kind = "prismt";
elseif any(arrayfun(@(n) istable(S.(n)), f))
    kind = "table";
elseif any(arrayfun(@(n) isSessionStruct(S.(n)), f))
    kind = "structs";
    var = f(find(arrayfun(@(n) isSessionStruct(S.(n)), f), 1));
elseif any(arrayfun(@(n) isRecordingArray(S.(n)), f))
    kind = "continuous";
    var = f(arrayfun(@(n) isRecordingArray(S.(n)), f))';
elseif any(~cellfun(@isempty, regexp(cellstr(f), '^dff_\d+$', 'once')))
    kind = "numbered";
else
    kind = "unknown";
end
end

function tf = isSessionStruct(v)
% A scalar struct whose fields include numbered sub-structs that hold a signal.
tf = false;
if ~isstruct(v) || ~isscalar(v), return; end
names = string(fieldnames(v));
numbered = names(~cellfun(@isempty, regexp(cellstr(names), '^[A-Za-z]\w*_\d+$', 'once')));
if isempty(numbered), return; end
d = v.(numbered(1));
tf = isstruct(d) && any(isfield(d, {'dff', 'zscore'}));
end

function tf = isRecordingArray(v)
% A struct array whose elements each hold a 2-D numeric recording (channels x time).
tf = false;
if ~isstruct(v) || isempty(v), return; end
tf = strlength(prismt.import.recordingField(v(1))) > 0;
end
