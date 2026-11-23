function run = hpo(cfg, opts)
%HPO Find the best model on this computer (nested selection).
%
%   cfg.hpo = struct('n_trials', 20);  run = prismt.hpo(cfg)
%   Settings are searched within each cross-validation fold, using only that fold's
%   training and validation trials; each fold's best settings are then retrained and
%   tested once. The pooled, honestly scored run is in the run folder's final/ subfolder
%   (prismt.loadResults opens it); hpo_summary.json lists the settings each fold chose
%   and the consensus settings.
arguments
    cfg struct
    opts.RunsFolder (1, 1) string = ""
    opts.Name (1, 1) string = ""
    opts.Wait (1, 1) logical = false
    opts.Python (1, 1) string = ""
end
run = prismt.train(cfg, RunsFolder=opts.RunsFolder, Name=opts.Name, Wait=opts.Wait, Python=opts.Python, Kind="hpo");
end
