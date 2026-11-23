function out = interpret(runDir, opts)
%INTERPRET Explain a finished classification run with PRISMt: attribution and motifs.
%
%   I = prismt.interpret(runDir)                  % waits, then returns prismt.loadInterpretation(runDir)
%   p = prismt.interpret(runDir, Wait=false)      % runs in the background (p.isRunning())
%   I = prismt.interpret(runDir, Classes=["CS-" "CS+"], Rank=3, Bootstrap=500)
%
%   Every held-out trial is attributed by the model of its own cross-validation fold: how
%   much each channel, time bin and signal pushed that trial's prediction towards one
%   class, routed through attention and carried by each token's value (the PRISMt
%   attribution). The attribution is reduced over the model's layers and heads and
%   factorized into a few motifs, each with a spatial profile (channels), a time course
%   and a share per signal. Results go to <runDir>/interpret/.
%
%   Options
%     Classes       the two classes to contrast (default: the run's two classes)
%     Rank          number of motifs, or "auto" (from the decomposition error)
%     Mode          "trials" (one expression per trial; default for recordings),
%                   "discriminative" (class difference; default for data with known ground truth)
%     Bootstrap     resamples for intervals (0 = none);  Permutations  for channel p-values
%     Compare       also compute the comparison methods (attention rollout, observation-level)
arguments
    runDir (1, 1) string
    opts.Classes (1, :) string = strings(1, 0)
    opts.Rank = "auto"
    opts.Mode (1, 1) string {mustBeMember(opts.Mode, ["auto", "trials", "discriminative"])} = "auto"
    opts.Bootstrap (1, 1) double {mustBeNonnegative, mustBeInteger} = 200
    opts.Permutations (1, 1) double {mustBeNonnegative, mustBeInteger} = 1000
    opts.Compare (1, 1) logical = true
    opts.Wait (1, 1) logical = true
    opts.Python (1, 1) string = ""
end
if ~isfile(fullfile(runDir, "config.json"))
    error('prismt:E_INTERP_MISSING', 'No PRISMT run at %s.', runDir);
end
args = ["interpret", runDir, "--rank", string(opts.Rank), "--mode", opts.Mode, ...
        "--bootstrap", string(opts.Bootstrap), "--permutations", string(opts.Permutations)];
if numel(opts.Classes) == 2, args = [args, "--classes", opts.Classes]; end
if ~opts.Compare, args(end + 1) = "--no-compare"; end
py = opts.Python;
if strlength(py) == 0, py = prismt.env.python(); end
if opts.Wait
    [res, status, log] = prismt.run.runPython(args, Python=py, Timeout=6 * 3600);
    if status ~= 0
        if isstruct(res) && isfield(res, 'error')
            e = res.error;
            error(sprintf('prismt:%s', char(e.code)), '%s: %s %s', e.title, e.message, e.hint);
        end
        error('prismt:E_INTERP_FAILED', 'PRISMt interpretation failed:\n%s', log);
    end
    out = prismt.loadInterpretation(runDir);
else
    folder = fullfile(runDir, "interpret");
    if ~isfolder(folder), mkdir(folder); end
    out = prismt.run.Process.start(py, ["-m", "prismt", args], Folder=prismt.internal.repoRoot(), ...
        Log=fullfile(folder, "log.txt"));
end
end
