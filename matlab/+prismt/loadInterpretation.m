function I = loadInterpretation(runDir)
%LOADINTERPRETATION What prismt.interpret wrote for a run (empty Summary if it has not run).
%
%   I = prismt.loadInterpretation(runDir)
%   I.Summary   interpret.json: classes, motifs (weight, modality call with interval,
%               stability, occlusion test), significant channels per signal, and for
%               data with known ground truth the recovery scores of every method
%   I.Mat       interpret.mat: motif_spatial (channels x K), motif_temporal (bins x K),
%               motif_modality (signals x K), motifs (K x channels x bins x signals),
%               motif_expression (trials x K, "trials" mode), region_p / region_q
%               (channels x signals), attribution maps, comparison maps
%   I.Status    interpret/status.json (state, message)
folder = fullfile(string(runDir), "interpret");
I = struct('Folder', folder, 'RunDir', string(runDir), 'Summary', [], 'Mat', struct(), 'Status', []);
if isfile(fullfile(folder, "status.json")), I.Status = jsondecode(fileread(fullfile(folder, "status.json"))); end
if isfile(fullfile(folder, "interpret.json")), I.Summary = jsondecode(fileread(fullfile(folder, "interpret.json"))); end
m = fullfile(folder, "interpret.mat");
if isfile(m)
    I.Mat = load(m);
    for f = ["channel_names", "modality_names", "classes", "method_names", "truth_names", "trial_subject", "trial_session"]
        if isfield(I.Mat, f)
            I.Mat.(f) = strtrim(string(I.Mat.(f)));
            I.Mat.(f) = I.Mat.(f)(:);
        end
    end
end
end
