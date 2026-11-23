function C = compareRuns(folders, opts)
%COMPARERUNS Put finished runs side by side: scores, intervals and paired differences.
%
%   C = prismt.compareRuns([runWithBehavior, runNeuralOnly])
%   disp(join(string(C.summary_lines), newline));  figure; prismt.plot.compareRuns(gca, C)
%
%   The first run is the reference. For classification runs tested on the same trials,
%   C.pairs gives each other run's difference to it per subject, with a 95% bootstrap
%   interval over subjects and a sign test: train with and without a signal (Task tab >
%   Which channels and signals), then compare, to ask what that signal adds.
arguments
    folders (1, :) string
    opts.Python (1, 1) string = ""
end
if numel(folders) < 2, error('prismt:E_COMPARE', 'Give at least two run folders to compare.'); end
[C, status, log] = prismt.run.runPython(["compare", folders], Python=opts.Python, Timeout=600);
if status ~= 0 || isempty(C)
    if isstruct(C) && isfield(C, 'error')
        error(sprintf('prismt:%s', char(C.error.code)), '%s: %s', C.error.title, C.error.message);
    end
    error('prismt:E_COMPARE', 'The runs could not be compared:\n%s', log);
end
end
