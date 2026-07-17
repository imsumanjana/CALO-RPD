# CALO-RPD Studio v3.2.0 Patch Notes

## Portfolio-first workflow

Portfolio Manager now follows Algorithm Selection and determines whether the study is a single-run diagnostic or an overall repeated experiment. Selected evidence drives the minimum number of paired runs, required stored fields, validation requirements, and unavailable-output warnings.

## Universal resume

- Campaign and run-job state is persisted before execution.
- Exact completed runs are reused using scientific fingerprints.
- Safe pause stops new admissions and commits active jobs before pausing.
- Interrupted jobs resume without repeating completed jobs.
- PPO training resumes from the last completed epoch.
- Bulk validation resumes from the remaining queue.
- Portfolio generation resumes artifact-by-artifact.
- Startup detects stale running records and offers Resume Center.

## Result portfolio engine

Selected evidence can generate convergence, feasibility, voltage, branch, control, scatter, heatmap, box, violin, CALO-diagnostic, statistical, caption, and reproducibility artifacts. Unsupported evidence is explicitly marked skipped instead of producing misleading empty plots.

## Scientific boundary

An active canonical optimizer job is not serialized mid-operator. Safe pause waits for it to finish; emergency stop restarts only that interrupted job from its original seed and frozen formulation. This preserves canonical algorithm semantics and reproducibility.
