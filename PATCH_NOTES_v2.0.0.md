# CALO-RPD Studio v2.0.0 Patch Notes

## Frozen CALO

The final benchmark workflow now verifies a cryptographic freeze manifest before TEST execution. The freeze covers the CALO Core v2 implementation, state and archive rules, PPO policy architecture and checkpoint, frozen historical training snapshot, CALO defaults, mixed-variable decoding, and constraint handling. Final benchmark TEST experiments are automatically locked out of historical learning.

## Full benchmark campaign

The new **Benchmark & Evidence** workspace creates a campaign over IEEE 30-, 57-, 118-, and 300-bus systems with all 20 primary algorithms, equal evaluation budgets, shared run-level seeds, and 30–50 independent runs per algorithm/task. Predefined study profiles cover deterministic ORPD, mixed discrete-continuous ORPD, load and renewable uncertainty, contingencies, mean-risk, worst-case, and CVaR.

## Statistics and evidence

Campaign evidence includes descriptive statistics, feasible-run rates, evaluations to first feasibility, runtime, Friedman testing, Holm-corrected Wilcoxon tests, Cliff's delta effect sizes, average ranks, and Nemenyi critical-difference information. The interpretation engine reports only evidence-supported statements and does not assert universal CALO superiority.

## Publication figures

The Transactions package can generate median convergence with IQR bands, feasibility probability, feasibility-attainment distributions, constraint decomposition, CALO operator usage/success, cognitive-regime timelines, objective boxplots and violin plots, ranking plots, critical-difference diagrams, and robustness maps.

## Research package

The package builder archives raw run JSON, seeds, final controls, full reconstructed solution states, convergence/final-population NPZ arrays, validation records, experiment configurations, frozen CALO source/checkpoint artifacts, statistical reports, publication figures, article-ready evidence summary, and a reproducibility ZIP.

## Important scientific boundary

v2.0.0 freezes and automates the final benchmark protocol. The release itself does **not** fabricate or pre-compute the final 30–50-run publication results. Those results must be generated on the user's hardware with the frozen TEST campaign and independently validated before article claims are finalized.
