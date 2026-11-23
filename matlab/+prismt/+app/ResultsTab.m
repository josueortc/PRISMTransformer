classdef ResultsTab < handle
    %RESULTSTAB Browse runs, read their summary, and plot their results.

    properties (SetAccess = private)
        App
        Runs
        RunTable = table()
        Summary
        PlotChoice
        ColorBy
        Options
        PlotArea
        Axes = {}
        Results = []          % loadResults of the open run
        RunDir string = ""
        Data = []             % the run's dataset (for colouring by trial columns), when available
        Interp = []           % prismt.loadInterpretation of the open run
        InterpButton
        InterpProcess = []
        InterpTimer = []
        Selected = zeros(1, 0)   % rows selected in the run list (several: compare them)
        Comparison = []          % prismt.compareRuns of the selected runs, while it is shown
        CompareButton
    end
    properties (Constant)
        ClassifyPlots = classifyPlots()
        MaePlots = ["Verdict", "Score vs reference models", "Learning curves", "Reconstruction example", ...
            "Predictability per channel", "Gain over baseline per channel", "Predictability by condition", ...
            "Summary space (embedding)"]
    end

    methods
        function t = ResultsTab(app, parent)
            t.App = app;
            g = prismt.app.ui.grid(parent, {'fit', '1x', 'fit'}, {420, '1x'});
            top = uigridlayout(g, [1 4], 'Padding', 0, 'ColumnWidth', {'1x', 'fit', 'fit', 'fit'}, 'RowHeight', {'fit'});
            top.BackgroundColor = g.BackgroundColor;
            prismt.app.ui.heading(top, "Runs");
            prismt.app.ui.button(top, "Refresh", @() app.safely(@t.refreshRuns));
            prismt.app.ui.button(top, "Open folder...", @() app.safely(@t.openOther), ...
                'Tooltip', "A run folder, or a cluster job folder with its results copied back");
            prismt.app.ui.button(top, "Watch", @() app.safely(@t.watchSelected), ...
                'Tooltip', "Follow the selected run on the Run tab (also after restarting MATLAB)");
            bar = uigridlayout(g, [1 6], 'Padding', 0, 'ColumnWidth', {'fit', 220, 'fit', 120, 130, 'fit'}, 'RowHeight', {'fit'});
            bar.BackgroundColor = g.BackgroundColor;
            uilabel(bar, 'Text', 'Show');
            t.PlotChoice = uidropdown(bar, 'Items', "-", 'ValueChangedFcn', @(~, ~) app.safely(@t.plotChanged));
            uilabel(bar, 'Text', 'By');
            t.ColorBy = uidropdown(bar, 'Items', "-", 'Tooltip', "Trial column to colour or group by", ...
                'ValueChangedFcn', @(~, ~) app.safely(@t.draw));
            t.Options = uidropdown(bar, 'Items', "-", 'Tooltip', "Signal, masking pattern or display option", ...
                'ValueChangedFcn', @(~, ~) app.safely(@t.draw));
            prismt.app.ui.button(bar, "Open in figure window", @() app.safely(@t.popOut));

            t.Runs = uitable(g, 'ColumnName', {'Run', 'Task', 'State', 'Result'}, 'RowName', {}, ...
                'ColumnWidth', {175, 55, 65, 'auto'}, 'Data', cell(0, 4), ...
                'CellSelectionCallback', @(~, e) app.safely(@() t.selected(e)));
            t.Runs.Layout.Row = 2; t.Runs.Layout.Column = 1;
            try   % several rows can be selected to compare runs (the property is newer than R2021a)
                t.Runs.Multiselect = 'on'; %#ok<*MCSUP> compat-ok
            catch
            end
            t.PlotArea = uigridlayout(g, [1 1], 'Padding', 0);
            t.PlotArea.BackgroundColor = g.BackgroundColor;
            t.PlotArea.Layout.Row = 2; t.PlotArea.Layout.Column = 2;
            left = uigridlayout(g, [2 1], 'Padding', 0, 'RowHeight', {215, 'fit'});
            left.BackgroundColor = g.BackgroundColor;
            left.Layout.Row = 3; left.Layout.Column = 1;
            t.Summary = uitextarea(left, 'Editable', 'off', 'Value', ["Choose a run in the list."; ""; ...
                "To compare runs (e.g. with and without a signal), select several with Ctrl/Cmd or Shift, then press Compare selected."]);
            b = uigridlayout(left, [3 2], 'Padding', 0, 'ColumnWidth', {'1x', '1x'}, 'RowHeight', {'fit', 'fit', 'fit'}, ...
                'ColumnSpacing', 6, 'RowSpacing', 5);
            b.BackgroundColor = g.BackgroundColor;
            t.InterpButton = prismt.app.ui.button(b, "Interpret (PRISMt)", @() app.safely(@() t.interpretRun(false)), ...
                'Tooltip', "Which channels, time bins and signals drive this classifier's predictions on held-out " + ...
                "trials, decomposed into motifs (the PRISMt method), each tested by removing it. Runs in the background; a few minutes.");
            t.CompareButton = prismt.app.ui.button(b, "Compare selected", @() app.safely(@t.compareSelected), ...
                'Tooltip', "Select two or more runs in the list (Ctrl/Cmd or Shift), then compare their scores. The first selected run is the reference.");
            prismt.app.ui.button(b, "Report (PDF)...", @() app.safely(@t.writeReport), ...
                'Tooltip', "A PDF with the verdict, the key figures, the settings and a methods paragraph for a manuscript");
            prismt.app.ui.button(b, "Export as MATLAB script...", @() app.safely(@t.exportScript), ...
                'Tooltip', "A script that reruns this analysis and draws these plots");
            prismt.app.ui.button(b, "Save figure...", @() app.safely(@t.saveFigure));
            prismt.app.ui.button(b, "Show folder", @() app.safely(@() prismt.app.ui.openFolder(t.RunDir)));
            n = prismt.app.ui.note(g, "Scores are from test trials the model never trained on. Reference models " + ...
                "show what simple methods achieve on the same trials; a result is interesting when it beats them.");
            n.Layout.Row = 3; n.Layout.Column = 2;
        end

        function refresh(t)
            t.refreshRuns();
        end

        function refreshRuns(t)
            T = prismt.listRuns();
            t.RunTable = T;
            if isempty(T) || height(T) == 0
                t.Runs.Data = cell(0, 4);
                return
            end
            t.Runs.Data = [cellstr(T.Name), cellstr(T.Task), cellstr(T.State), cellstr(T.Summary)];
        end

        function openRun(t, folder)
            %OPENRUN Load a run (or a cluster job folder) and show its first plot.
            folder = string(folder);
            h = t.App.Dialogs.busy("Loading results...");
            cleanup = onCleanup(@() delete(h));
            R = prismt.loadResults(folder);
            if isempty(R.Metrics)
                st = "";
                if ~isempty(R.Status), st = string(R.Status.state); end
                error('prismt:app', 'This run has no results yet (state: %s). Finished runs have a metrics.json.', st);
            end
            t.Results = R;
            t.RunDir = string(R.Folder);
            t.Comparison = [];
            t.Data = [];
            t.Interp = prismt.loadInterpretation(t.RunDir);
            t.InterpButton.Enable = onoff(string(R.Config.task) == "classify");
            try
                f = string(R.Config.dataset.path);
                if isfile(f), t.Data = prismt.loadDataset(f); end
            catch
            end
            lines = string(R.Metrics.summary_lines);
            if isfield(R.Metrics, 'verdict') && ~isempty(R.Metrics.verdict)
                v = R.Metrics.verdict;
                if iscell(v), v = [v{:}]; end
                marks = struct('good', "OK", 'caution', "CHECK", 'bad', "PROBLEM", 'info', "NOTE");
                vl = strings(numel(v), 1);
                for k = 1:numel(v)
                    vl(k) = "[" + marks.(char(v(k).status)) + "] " + string(v(k).question) + " " + string(v(k).answer) + ".";
                end
                lines = [vl; ""; lines(:)];
            end
            if isfield(R.Metrics, 'warnings') && ~isempty(R.Metrics.warnings)
                w = R.Metrics.warnings;
                if iscell(w), w = [w{:}]; end
                if isstruct(w), lines = [lines; ""; "Warnings:"; "- " + string({w.message})']; end
            end
            t.Summary.Value = [lines; ""; "Folder: " + folder];
            task = string(R.Config.task);
            if task == "mae", t.PlotChoice.Items = t.MaePlots; else, t.PlotChoice.Items = t.ClassifyPlots; end
            t.PlotChoice.Value = t.PlotChoice.Items(1);
            t.plotChanged();
            t.addMotifSummary();
        end

        function show(t, name)
            %SHOW Pick a plot by its menu name (used by tests).
            t.PlotChoice.Value = name;
            t.plotChanged();
        end

        function C = compareSelected(t, rows)
            %COMPARESELECTED Compare the runs selected in the list (or the given rows of it).
            if nargin < 2, rows = t.Selected; end
            rows = rows(rows <= height(t.RunTable));
            if numel(rows) < 2
                error('prismt:app', 'Select two or more finished runs in the list first (Ctrl/Cmd or Shift and click).');
            end
            h = t.App.Dialogs.busy("Comparing runs...");
            cleanup = onCleanup(@() delete(h));
            C = prismt.compareRuns(t.RunTable.Folder(rows)');
            t.Comparison = C;
            t.Summary.Value = [string(C.summary_lines(:)); ""; "The first selected run is the reference for the differences."];
            items = string(t.PlotChoice.Items);
            items = [items(items ~= "-" & items ~= "Runs compared"), "Runs compared"];
            t.PlotChoice.Items = items;
            t.PlotChoice.Value = "Runs compared";
            t.ColorBy.Enable = 'off'; t.Options.Enable = 'off';
            delete(t.PlotArea.Children);
            t.PlotArea.ColumnWidth = {'1x'};
            t.Axes = {uiaxes(t.PlotArea)};
            prismt.plot.compareRuns(t.Axes{1}, C);
        end

        function file = writeReport(t)
            %WRITEREPORT A PDF report of the open run (asks where to save it).
            if isempty(t.Results), error('prismt:app', 'Open a run first.'); end
            file = t.App.Dialogs.putFile({'*.pdf', 'PDF report (*.pdf)'}, "Save the report", ...
                fullfile(t.RunDir, "report.pdf"));
            if strlength(file) == 0, return; end
            h = t.App.Dialogs.busy("Writing the report...");
            cleanup = onCleanup(@() delete(h));
            file = prismt.report(t.RunDir, File=file, Open=~t.App.Dialogs.Headless);
        end

        function interpretRun(t, wait)
            %INTERPRETRUN Run PRISMt on the open run (in the background unless wait is true).
            if isempty(t.Results), error('prismt:app', 'Open a classification run first.'); end
            if string(t.Results.Config.task) ~= "classify"
                error('prismt:app', 'PRISMt explains classifications; this run is a masked autoencoder.');
            end
            if wait
                h = t.App.Dialogs.busy("Interpreting with PRISMt...");
                cleanup = onCleanup(@() delete(h));
                t.Interp = prismt.interpret(t.RunDir);
                t.show("PRISMt motifs");
                t.addMotifSummary();
                return
            end
            t.InterpProcess = prismt.interpret(t.RunDir, Wait=false);
            t.InterpButton.Enable = 'off';
            t.InterpButton.Text = "Interpreting...";
            t.InterpTimer = timer('ExecutionMode', 'fixedSpacing', 'Period', 2, ...
                'TimerFcn', @(~, ~) t.pollInterpretation(), 'ErrorFcn', @(~, ~) []);
            start(t.InterpTimer);
        end

        function file = exportScript(t)
            if isempty(t.Results), error('prismt:app', 'Open a run first.'); end
            file = t.App.Dialogs.putFile({'*.m', 'MATLAB script (*.m)'}, "Export as MATLAB script", ...
                fullfile(prismt.internal.projectFolder(), "scripts", "prismt_" + string(t.Results.Config.task) + "_run.m"));
            if strlength(file) == 0, return; end
            cfg = runConfig(t.Results.TopFolder, t.Results.Config);
            t.App.Controller.exportScript(file, cfg);
            if ~t.App.Dialogs.Headless, edit(char(file)); end
        end

        function saveFigure(t)
            if isempty(t.Axes), return; end
            file = t.App.Dialogs.putFile({'*.png;*.pdf;*.svg', 'Image (*.png, *.pdf, *.svg)'}, "Save figure", ...
                "prismt_" + replace(lower(string(t.PlotChoice.Value)), " ", "_") + ".png");
            if strlength(file) == 0, return; end
            f = t.popOut(false);
            exportgraphics(f, file, 'Resolution', 200);
            close(f);
        end

        function f = popOut(t, visible)
            %POPOUT Redraw the current plot in a normal figure window (to edit, zoom or save).
            if nargin < 2, visible = true; end
            if isempty(t.Results), f = []; return; end
            f = figure('Color', 'w', 'Visible', onoff(visible), 'Name', "PRISMT: " + t.PlotChoice.Value);
            n = numel(t.Axes);
            if n > 1
                tl = tiledlayout(f, 1, n, 'TileSpacing', 'compact');
                axs = arrayfun(@(~) nexttile(tl), 1:n, 'UniformOutput', false);
            else
                axs = {axes(f)};
            end
            t.drawInto(axs);
        end
    end

    methods (Access = private)
        function pollInterpretation(t)
            if ~isempty(t.InterpProcess) && t.InterpProcess.isRunning()
                st = prismt.loadInterpretation(t.RunDir).Status;
                if ~isempty(st) && isfield(st, 'message'), t.InterpButton.Text = string(st.message); end
                return
            end
            stop(t.InterpTimer); delete(t.InterpTimer); t.InterpTimer = [];
            t.InterpButton.Enable = 'on';
            t.InterpButton.Text = "Interpret (PRISMt)";
            t.Interp = prismt.loadInterpretation(t.RunDir);
            st = t.Interp.Status;
            if ~isempty(st) && string(st.state) == "failed"
                e = st.error;
                t.App.Dialogs.alert(string(e.message) + " " + string(e.hint), string(e.title));
                return
            end
            t.show("PRISMt motifs");
            t.addMotifSummary();
        end

        function addMotifSummary(t)
            % What PRISMt found, in words, appended to the run's summary.
            I = t.Interp;
            if isempty(I) || isempty(I.Summary) || ~isfield(I.Summary, 'motifs'), return; end
            mm = I.Summary.motifs;
            if iscell(mm), mm = [mm{:}]; end
            lines = "PRISMt (" + string(I.Summary.classes{1}) + " vs " + string(I.Summary.classes{2}) + "): " + numel(mm) + " motifs.";
            for k = 1:numel(mm)
                if isfield(mm(k), 'test') && isstruct(mm(k).test) && isfield(mm(k).test, 'answer')
                    lines(end + 1) = "Motif " + k + ": " + string(mm(k).test.answer); %#ok<AGROW>
                end
            end
            keep = t.Summary.Value;
            cut = find(startsWith(string(keep), "PRISMt ("), 1);
            if ~isempty(cut), keep = keep(1:cut - 1); end
            t.Summary.Value = [string(keep(:)); lines(:)];
        end

        function selected(t, e)
            if isempty(e.Indices), return; end
            rows = unique(e.Indices(:, 1), 'stable')';
            rows = rows(rows <= height(t.RunTable));
            t.Selected = rows;
            if numel(rows) == 1
                t.openRun(t.RunTable.Folder(rows));
            elseif numel(rows) > 1
                t.Summary.Value = numel(rows) + " runs selected. Press Compare selected to put their scores side by side.";
            end
        end

        function openOther(t)
            f = t.App.Dialogs.getFolder("Choose a run folder or a cluster job folder", prismt.internal.projectFolder());
            if strlength(f) == 0, return; end
            t.openRun(f);
        end

        function watchSelected(t)
            if strlength(t.RunDir) == 0, error('prismt:app', 'Choose a run in the list first.'); end
            t.App.Controller.watch(t.RunDir);
            t.App.show("Run");
        end

        function plotChanged(t)
            name = string(t.PlotChoice.Value);
            if name == "Runs compared" && ~isempty(t.Comparison)
                delete(t.PlotArea.Children);
                t.PlotArea.ColumnWidth = {'1x'};
                t.Axes = {uiaxes(t.PlotArea)};
                t.draw();
                return
            end
            t.Comparison = [];
            R = t.Results;
            cols = strings(1, 0);
            if ~isempty(t.Data), cols = prismt.app.ui.columnChoices(t.Data); end
            opts = "-";
            needs = false;
            switch name
                case "Summary space (embedding)"
                    needs = true;
                    if isempty(cols) && isfield(R.Mat, 'class_names'), cols = "true class"; end
                    if isfield(R.Mat, 'embedding_fold') && max(R.Mat.embedding_fold) > 1
                        opts = "Fold " + (1:max(R.Mat.embedding_fold));
                    end
                case "Predictability by condition"
                    needs = true;
                    opts = maskNames(R);
                case {"Predictability per channel", "Gain over baseline per channel", "Reconstruction example"}
                    opts = modalityNames(R);
                case "Confusion matrix"
                    opts = ["Fractions", "Counts"];
                case {"PRISMt motifs", "PRISMt motif expression", "PRISMt motifs on the map"}
                    if hasInterp(t.Interp), opts = "Motif " + (1:size(t.Interp.Mat.motifs, 1)); end
                    if name == "PRISMt motif expression"
                        cols = ["class", "subject", "session"];
                        needs = true;
                    end
                case "PRISMt channels"
                    if hasInterp(t.Interp), opts = t.Interp.Mat.modality_names(:)'; end
                case "Learning curves"
                    opts = ["Loss", "Validation score"];
            end
            if needs && isempty(cols)
                cols = "(dataset not found)";
            end
            if isempty(cols), cols = "-"; end
            t.ColorBy.Items = cols;
            t.ColorBy.Enable = onoff(needs && cols(1) ~= "(dataset not found)");
            if startsWith(name, "PRISMt") && name ~= "PRISMt motif expression", t.ColorBy.Enable = 'off'; end
            if needs && ~isempty(t.Data) && ~startsWith(name, "PRISMt")
                pick = intersect(["phase", string(t.Data.Subject), "stim"], cols, 'stable');
                if ~isempty(pick), t.ColorBy.Value = pick(1); end
            end
            t.Options.Items = opts;
            t.Options.Enable = onoff(opts(1) ~= "-");
            nAxes = 1;
            if name == "Reconstruction example", nAxes = 4; end
            if name == "PRISMt motifs on the map" && hasInterp(t.Interp)
                nAxes = min(numel(t.Interp.Mat.modality_names), 4);
            end
            delete(t.PlotArea.Children);
            t.PlotArea.ColumnWidth = repmat({'1x'}, 1, nAxes);
            t.Axes = cell(1, nAxes);
            for k = 1:nAxes, t.Axes{k} = uiaxes(t.PlotArea); end
            t.draw();
        end

        function draw(t)
            if isempty(t.Results) || isempty(t.Axes), return; end
            for k = 1:numel(t.Axes), cla(t.Axes{k}, 'reset'); end
            t.drawInto(t.Axes);
        end

        function drawInto(t, axs)
            if string(t.PlotChoice.Value) == "Runs compared"
                if ~isempty(t.Comparison), prismt.plot.compareRuns(axs{1}, t.Comparison); end
                return
            end
            R = t.Results;
            name = string(t.PlotChoice.Value);
            opt = string(t.Options.Value);
            ax = axs{1};
            switch name
                case "Verdict", prismt.plot.verdict(ax, R);
                case "Score vs reference models", prismt.plot.scoreVsBaselines(ax, R);
                case "Learning curves"
                    if opt == "Validation score"
                        metric = "val_balanced_accuracy";
                        if string(R.Config.task) == "mae", metric = "val_r2"; end
                        prismt.plot.learningCurves(ax, R, Metric=metric);
                    else
                        prismt.plot.learningCurves(ax, R);
                    end
                case "Confusion matrix", prismt.plot.confusion(ax, R, Normalize=(opt ~= "Counts"));
                case "Accuracy per subject", prismt.plot.accuracyByGroup(ax, R, "subject");
                case "Accuracy per session", prismt.plot.accuracyByGroup(ax, R, "session");
                case "Confidence", prismt.plot.probabilities(ax, R);
                case "Reconstruction example", prismt.plot.reconExample([axs{:}], R, Modality=opt);
                case "Predictability per channel", prismt.plot.r2Channels(ax, R, Modality=opt);
                case "Gain over baseline per channel", prismt.plot.r2Channels(ax, R, Modality=opt, VersusBaseline=true);
                case "Predictability by condition"
                    if isempty(t.Data), noData(ax); return; end
                    prismt.plot.r2ByCondition(ax, R, t.Data, string(t.ColorBy.Value), Mask=opt);
                case "Summary space (embedding)"
                    by = t.colorValues();
                    if isempty(by), noData(ax); return; end
                    fold = 1;
                    if startsWith(opt, "Fold "), fold = str2double(extractAfter(opt, "Fold ")); end
                    prismt.plot.embedding(ax, R, by, Fold=fold);
                otherwise
                    if startsWith(name, "PRISMt")
                        t.drawInterpretation(axs, name, opt);
                    end
            end
        end

        function drawInterpretation(t, axs, name, opt)
            ax = axs{1};
            if ~hasInterp(t.Interp)
                prismt.plot.prepareAxes(ax);
                title(ax, "Not interpreted yet: press Interpret (PRISMt) below", 'FontWeight', 'normal');
                axis(ax, 'off');
                return
            end
            I = t.Interp;
            k = 1;
            if startsWith(opt, "Motif "), k = str2double(extractAfter(opt, "Motif ")); end
            switch name
                case "PRISMt motifs", prismt.plot.motifMaps(ax, I, Motif=k);
                case "PRISMt motifs on the map", prismt.plot.motifBrain(axs, I, Motif=k, Dataset=t.Data);
                case "PRISMt motif test", prismt.plot.motifTest(ax, I);
                case "PRISMt signal shares", prismt.plot.modalityAllocation(ax, I);
                case "PRISMt motif expression"
                    by = string(t.ColorBy.Value);
                    if ~ismember(by, ["class", "subject", "session"]), by = "class"; end
                    if (by == "subject" && ~isfield(I.Mat, 'trial_subject')) || (by == "session" && ~isfield(I.Mat, 'trial_session'))
                        by = "class";
                    end
                    prismt.plot.motifExpression(ax, I, Motif=k, By=by);
                case "PRISMt channels", prismt.plot.attributionRegions(ax, I, Modality=opt);
            end
        end

        function by = colorValues(t)
            by = [];
            col = string(t.ColorBy.Value);
            if ~isempty(t.Data) && ismember(col, string(t.Data.Trials.Properties.VariableNames))
                by = prismt.app.ui.labels(t.Data, col);
            end
        end
    end
end

function names = classifyPlots()
names = ["Verdict", "Score vs reference models", "Learning curves", "Confusion matrix", ...
    "Accuracy per subject", "Accuracy per session", "Confidence", "Summary space (embedding)", ...
    "PRISMt motifs", "PRISMt motifs on the map", "PRISMt motif test", "PRISMt signal shares", ...
    "PRISMt motif expression", "PRISMt channels"];
end

function cfg = runConfig(folder, resolved)
% The settings as given (run.json), else the resolved ones.
f = fullfile(folder, "run.json");
if isfile(f), cfg = jsondecode(fileread(f)); else, cfg = resolved; end
end

function names = maskNames(R)
names = "-";
if isfield(R.Mat, 'mask_names') && ~isempty(R.Mat.mask_names), names = R.Mat.mask_names(:)'; end
end

function names = modalityNames(R)
names = "-";
if isfield(R.Mat, 'modality_names') && ~isempty(R.Mat.modality_names), names = R.Mat.modality_names(:)'; end
end

function tf = hasInterp(I)
tf = ~isempty(I) && isstruct(I) && isfield(I, 'Mat') && isfield(I.Mat, 'motifs');
end

function noData(ax)
prismt.plot.prepareAxes(ax);
title(ax, "The run's dataset was not found on this computer", 'FontWeight', 'normal');
axis(ax, 'off');
end

function s = onoff(tf)
if tf, s = 'on'; else, s = 'off'; end
end
