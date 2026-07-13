# CALO-RPD Studio 1.0.10

## Experiment Manager workflow and layout correction

The Experiment Manager now follows the visible order required by the scientific workflow:

1. **Experiment configuration**
2. **Fairness audit**
3. **Run study**
4. **Run queue**

The primary-comparison and CALO-ablation buttons remain disabled until the fairness audit passes for the current configuration. Any configuration change invalidates the previous audit and locks execution again.

The Experiment Manager is now the fourth genuinely long workspace to use a page-level vertical scroll area. The workspace header remains fixed, horizontal scrolling is disabled, and the workflow body scrolls only when the available height is insufficient. This prevents Qt from vertically compressing numeric controls, combo boxes, text, and action buttons on laptop-height windows.

No scientific optimization, power-flow, fairness, or result semantics were changed in this release.
