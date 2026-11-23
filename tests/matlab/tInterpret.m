classdef (TestTags = {'Python'}) tInterpret < matlab.unittest.TestCase
    %TINTERPRET PRISMt from MATLAB: interpret a run, load it, draw every plot.
    %   Needs PRISMT_PYTHON (a Python with PRISMT's dependencies).

    properties
        Dir
        RunDir
    end

    methods (TestClassSetup)
        function setup(tc)
            tc.assumeTrue(isfile(getenv('PRISMT_PYTHON')), 'Set PRISMT_PYTHON to run PRISMt tests.');
            tc.Dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
            setenv('PRISMT_SETTINGS_DIR', fullfile(tc.Dir, 'settings'));
            ds = prismt.demo.makeSyntheticDataset(Profile="tiny");
            data = prismt.writeDataset(ds, fullfile(tc.Dir, "tiny.mat"));
            cfg = prismt.defaultConfig("classify", data, Label="stim", RunsFolder=fullfile(tc.Dir, "runs"));
            cfg.train = struct('epochs', 3, 'min_epochs', 0, 'device', 'cpu');
            cfg.model = struct('d_model', 16, 'n_layers', 2, 'n_heads', 2);
            run = prismt.train(cfg, Wait=true);
            tc.RunDir = run.RunDir;
        end
    end

    methods (Test)
        function interpretAndDrawEveryPlot(tc)
            I = prismt.interpret(tc.RunDir, Rank=2, Bootstrap=3, Permutations=20, Python=string(getenv('PRISMT_PYTHON')));
            tc.verifyEqual(size(I.Mat.motifs, 1), 2);
            tc.verifyEqual(sum(I.Mat.motif_modality, 1), [1 1], 'AbsTol', 1e-6);
            tc.verifyEqual(I.Mat.classes, ["CS-"; "CS+"]);
            tc.verifyTrue(isfield(I.Summary, 'significant_regions'));
            f = figure('Visible', 'off');
            c = onCleanup(@() close(f));
            prismt.plot.motifMaps(axes(f), I, Motif=2);
            prismt.plot.modalityAllocation(axes(f), I);
            prismt.plot.motifExpression(axes(f), I, By="subject");
            prismt.plot.attributionRegions(axes(f), I, Modality=I.Mat.modality_names(1));
            prismt.plot.motifTest(axes(f), I);
            ds = prismt.demo.makeSyntheticDataset(Profile="tiny");
            prismt.plot.motifBrain({axes(f), axes(f)}, I, Motif=1, Dataset=ds);
            prismt.plot.motifBrain({axes(f), axes(f)}, I, Motif=2);          % no layout: bars
            tc.verifyEqual(numel(I.Mat.motif_test_cost), 2);
            tc.verifyTrue(isfield(I.Summary.motifs(1).test, 'answer'));
            again = prismt.loadInterpretation(tc.RunDir);
            tc.verifyEqual(string(again.Status.state), "finished");
        end

        function verdictComparisonAndReport(tc)
            py = string(getenv('PRISMT_PYTHON'));
            R = prismt.loadResults(tc.RunDir);
            keys = string({R.Metrics.verdict.key});
            tc.verifyEqual(keys(end), "trained");
            tc.verifyTrue(all(ismember(string({R.Metrics.verdict.status}), ["good", "caution", "bad", "info"])));
            f = figure('Visible', 'off');
            c = onCleanup(@() close(f));
            prismt.plot.verdict(axes(f), R);
            prismt.plot.scoreVsBaselines(axes(f), R);
            C = prismt.compareRuns([tc.RunDir, tc.RunDir], Python=py);
            tc.verifyEqual(numel(C.runs), 2);
            tc.verifyEqual(C.pairs(1).difference, 0, 'AbsTol', 1e-12);
            prismt.plot.compareRuns(axes(f), C);
            file = prismt.report(tc.RunDir, File=fullfile(tc.Dir, "report.pdf"), Python=py);
            tc.verifyTrue(isfile(file));
            tc.verifyTrue(isfile(fullfile(tc.Dir, "report_methods.txt")));
        end

        function interpretInTheBackground(tc)
            p = prismt.interpret(tc.RunDir, Rank=1, Bootstrap=0, Permutations=10, Compare=false, Wait=false, ...
                Python=string(getenv('PRISMT_PYTHON')));
            code = p.wait(600);
            tc.verifyEqual(code, 0);
            I = prismt.loadInterpretation(tc.RunDir);
            tc.verifyEqual(size(I.Mat.motifs, 1), 1);
        end

    end
end
