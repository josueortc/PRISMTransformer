function file = report(runDir, opts)
%REPORT Write a PDF report of a run: verdict, figures, settings and a methods paragraph.
%
%   file = prismt.report(runDir)                 % <runDir>/report.pdf (and report_methods.txt)
%   file = prismt.report(runDir, File="my_report.pdf", Open=true)
%
%   Page 1 answers the questions in words; the next pages show the figures behind the
%   answers (and the PRISMt motifs, if the run was interpreted); the last page lists the
%   settings and a methods paragraph to paste into a manuscript and edit.
arguments
    runDir (1, 1) string
    opts.File (1, 1) string = ""
    opts.Open (1, 1) logical = false
    opts.Python (1, 1) string = ""
end
args = ["report", runDir];
if strlength(opts.File), args = [args, "--out", opts.File]; end
[res, status, log] = prismt.run.runPython(args, Python=opts.Python, Timeout=600);
if status ~= 0 || isempty(res)
    if isstruct(res) && isfield(res, 'error')
        error(sprintf('prismt:%s', char(res.error.code)), '%s: %s %s', res.error.title, res.error.message, res.error.hint);
    end
    error('prismt:E_REPORT', 'The report could not be written:\n%s', log);
end
file = string(res.report);
if opts.Open, open(char(file)); end
end
