classdef (TestTags = {'Python'}) tBackend < matlab.unittest.TestCase
    %TBACKEND MATLAB -> Python -> MATLAB: environment, runs in the background, results, jobs.
    %   Needs PRISMT_PYTHON (a Python with PRISMT's dependencies).

    properties
        Dir
        Data
    end

    methods (TestClassSetup)
        function setup(tc)
            tc.assumeTrue(isfile(getenv('PRISMT_PYTHON')), 'Set PRISMT_PYTHON to run backend tests.');
            tc.Dir = string(tc.applyFixture(matlab.unittest.fixtures.TemporaryFolderFixture).Folder);
            setenv('PRISMT_SETTINGS_DIR', fullfile(tc.Dir, 'settings'));
            ds = prismt.demo.makeSyntheticDataset(Profile="tiny");
            tc.Data = prismt.writeDataset(ds, fullfile(tc.Dir, "tiny.mat"));
        end
    end

    methods (Test)
        function doctorAcceptsTheEnvironment(tc)
            r = prismt.env.doctor(string(getenv('PRISMT_PYTHON')));
            tc.verifyTrue(r.ok);
            tc.verifyTrue(ismember(string(r.device), ["cpu", "mps", "cuda"]));
            bad = prismt.env.doctor("/usr/bin/false");
            tc.verifyFalse(bad.ok);
        end

        function childEnvironmentIsClean(tc)
            env = prismt.run.Process.cleanEnvironment(string(getenv('PRISMT_PYTHON')), struct());
            keys = string(env.keys());
            tc.verifyFalse(any(startsWith(keys, ["KMP_", "GFORTRAN_"])));
            tc.verifyFalse(any(keys == "MATLABPATH"));
            tc.verifyEqual(string(env('PYTHONPATH')), string(fullfile(prismt.internal.repoRoot(), 'src')));
            tc.verifyFalse(contains(string(env('PATH')), matlabroot));
        end

        function checkReportsPlanAndProblems(tc)
            cfg = prismt.defaultConfig("classify", tc.Data, Label="stim");
            r = prismt.check(cfg);
            tc.verifyTrue(r.ok);
            tc.verifyEqual(r.selection.class_counts(:)', [64 64]);
            cfg.labels.column = 'nope';
            r = prismt.check(cfg);
            tc.verifyFalse(r.ok);
            tc.verifyEqual(string(r.error.code), "E_SEL_COLUMN");
            tc.verifySubstring(string(r.error.hint), "stim");
        end

        function backgroundRunFinishesAndLoads(tc)
            cfg = prismt.defaultConfig("classify", tc.Data, Label="stim", RunsFolder=fullfile(tc.Dir, "runs"));
            cfg.train = struct('epochs', 3, 'min_epochs', 0, 'device', 'cpu');
            t0 = tic;
            run = prismt.train(cfg);
            tc.verifyLessThan(toc(t0), 10, "prismt.train must return at once");
            s = run.wait(300);
            tc.verifyEqual(string(s.state), "finished");
            R = prismt.loadResults(run.RunDir);
            tc.verifyEqual(R.Mat.class_names, ["CS-"; "CS+"]);
            tc.verifyGreaterThan(height(R.History), 0);
            tc.verifyTrue(isfile(fullfile(run.RunDir, "launch.sh")) || isfile(fullfile(run.RunDir, "launch.cmd")));
            T = prismt.listRuns(fullfile(tc.Dir, "runs"));
            tc.verifyEqual(height(T), 1);
        end

        function stopEndsARunGracefully(tc)
            cfg = prismt.defaultConfig("mae", tc.Data, RunsFolder=fullfile(tc.Dir, "runs"));
            cfg.train = struct('epochs', 500, 'min_epochs', 500, 'device', 'cpu');
            run = prismt.train(cfg);
            t0 = tic;
            while toc(t0) < 120 && ~(isfield(run.status(), 'epoch') && run.status().epoch >= 2), pause(1); end
            run.stop();
            s = run.wait(120);
            tc.verifyEqual(string(s.state), "finished");
            tc.verifyTrue(isfile(fullfile(run.RunDir, "metrics.json")), "the best model so far is evaluated");
        end

        function failureIsExplained(tc)
            cfg = prismt.defaultConfig("classify", fullfile(tc.Dir, "missing.mat"), Label="stim", ...
                RunsFolder=fullfile(tc.Dir, "runs"));
            run = prismt.train(cfg);
            s = run.wait(120);
            tc.verifyEqual(string(s.state), "failed");
            tc.verifyTrue(strlength(string(s.error.hint)) > 0);
        end

        function clusterJobFolderHasEverything(tc)
            cfg = prismt.defaultConfig("classify", tc.Data, Label="stim");
            job = prismt.makeClusterJob(cfg, struct('gpus', 0, 'host', 'cluster.example.edu', 'user', 'me'), ...
                Output=fullfile(tc.Dir, "cluster"), RemoteDataset="~/data/tiny.mat");
            tc.verifyTrue(isfile(fullfile(job.Folder, "submit.sh")));
            tc.verifySubstring(job.Readme, "me@cluster.example.edu");
            tc.verifyTrue(isfolder(fullfile(job.Folder, "logs")));
        end

        function importsATableOfSessions(tc)
            [ds, ~] = prismt.demo.makeSyntheticDataset(Profile="tiny");
            % A table with one row per session.
            key = string(ds.Trials.mouse) + "/" + ds.Trials.session;
            [u, ~, j] = unique(key, 'stable');
            dff = cell(numel(u), 1); stim = cell(numel(u), 1); speed = cell(numel(u), 1);
            mouse = strings(numel(u), 1); phase = strings(numel(u), 1);
            for k = 1:numel(u)
                rows = j == k;
                dff{k} = double(ds.X(rows, :, :, 1));             % trials x channels x time
                stim{k} = ds.Trials.stim(rows);
                speed{k} = rand(nnz(rows), ds.T);
                mouse(k) = string(ds.Trials.mouse(find(rows, 1)));
                phase(k) = string(ds.Trials.phase(find(rows, 1)));
            end
            T = table(dff, stim, speed, cellstr(mouse), cellstr(phase), 'VariableNames', {'dff', 'stim', 'runSpeed', 'mouse', 'phase'});
            meta = struct('frameRateHz', 10, 'windowSeconds', [ds.Times(1) ds.Times(end)], 'stimCoding', '0=CS-, 1=CS+'); %#ok<NASGU>
            f = fullfile(tc.Dir, "table.mat");
            save(f, 'T', 'meta');
            [out, rep] = prismt.importData(f, Behavior="runSpeed");
            tc.verifyEqual(out.N, ds.N);
            tc.verifyEqual(out.X(:, 1:ds.R, :, 1), ds.X(:, :, :, 1), 'AbsTol', 1e-5);
            tc.verifyEqual(out.ModalityNames, ["dff"; "behavior"]);
            tc.verifyEqual(out.ChannelNames(end), "runSpeed");
            tc.verifyEqual(out.ValueLabels.Label(out.ValueLabels.Column == "stim"), ["CS-"; "CS+"]);
            tc.verifyEqual(out.Times, ds.Times, 'AbsTol', 1e-9);
            tc.verifyNotEmpty(rep.notes);
            tc.verifyEmpty(out.validate());
            % the result is trainable
            g = prismt.writeDataset(out, fullfile(tc.Dir, "imported.mat"));
            r = prismt.check(prismt.defaultConfig("classify", g, Label="stim"));
            tc.verifyTrue(r.ok);
        end

        function importsSessionStructsAndContinuousRecordings(tc)
            % one numbered sub-struct per session, whatever the variable is called
            rng(1);
            sessions = struct();
            for k = 1:3
                sessions.(sprintf('dataset_%03d', k)) = struct('dff', rand(5, 7, 4), 'stim', (1:5)' > 2, ...
                    'mouse', sprintf('M%d', k), 'phase', 'early');
            end
            f = fullfile(tc.Dir, "structs.mat");
            S = struct('anything', sessions); %#ok<NASGU>
            save(f, '-struct', 'S');
            [kind, var] = prismt.import.detect(load(f));
            tc.verifyEqual(kind, "structs");
            tc.verifyEqual(var, "anything");
            ds = prismt.importData(f);
            tc.verifyEqual([ds.N ds.R ds.T], [15 4 7]);        % trials x time x channels by default
            tc.verifyEmpty(ds.validate());

            % continuous recordings: one struct array per group, one element per subject
            a = struct('data', {rand(6, 95), rand(6, 120)}, 'mouse_id', {'a1', 'a2'});
            b = struct('data', {rand(6, 100)}, 'mouse_id', {'b1'});
            C = struct('rec_ctl_struct', a, 'rec_trt_struct', b); %#ok<NASGU>
            g = fullfile(tc.Dir, "continuous.mat");
            save(g, '-struct', 'C');
            [kind, vars] = prismt.import.detect(load(g));
            tc.verifyEqual(kind, "continuous");
            tc.verifyEqual(sort(vars), ["rec_ctl_struct", "rec_trt_struct"]);
            ds = prismt.importData(g, Window=30);
            tc.verifyEqual(ds.N, 3 + 4 + 3);                   % floor(95/30) + floor(120/30) + floor(100/30)
            tc.verifyEqual(ds.R, 6);
            tc.verifyEqual(sort(unique(string(ds.Trials.group)))', ["ctl", "trt"]);
            tc.verifyEqual(numel(unique(string(ds.Trials.mouse))), 3);
        end

        function mapsUseALabelImageOrPositionsWithoutAnAtlas(tc)
            f = figure('Visible', 'off');
            c = onCleanup(@() close(f));
            labels = kron([1 2; 3 4], ones(5));                 % 4 channels as a label image
            prismt.plot.channelMap(axes(f), [1 2 3 4]', Labels=labels, Highlight=[true false false false]');
            prismt.plot.channelMap(axes(f), [1 2 3 4]', X=[1 2 1 2]', Y=[1 1 2 2]', Names="c" + (1:4)');
            prismt.plot.channelMap(axes(f), [1 2 3 4]', Atlas="no-such-atlas", Names="c" + (1:4)');   % bars
            tc.verifyEmpty(prismt.internal.atlas("no-such-atlas", 4));
        end

        function windowingDoesNotInterleave(tc)
            X = (1:100)' + 1000 * (1:3);                       % time x channels, X(t,r) = t + 1000r
            W = prismt.import.windowContinuous(X, 30, 20);
            tc.verifySize(W, [4 3 30]);
            tc.verifyEqual(squeeze(W(2, 3, :))', (21:50) + 3000);   % window 2 starts at sample 21
        end
    end
end
