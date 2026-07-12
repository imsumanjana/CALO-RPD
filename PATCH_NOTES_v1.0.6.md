# CALO-RPD Studio v1.0.6

## Results review workflow correction

- The Results Explorer review action now performs a visible workflow transition.
- After selecting a run and clicking **Confirm result review and continue to validation**, the workflow marks the review as complete, unlocks Validation & Audit, transfers the selected experiment/run, and opens Validation & Audit automatically.
- The validation page opens with the reviewed run already selected.

## Selective plot export

- The **Export figure** popup now discovers the legend-capable series currently present in the preview.
- Each available preview series is shown as a checkbox using its displayed legend name.
- Users can select any subset of curves before saving.
- The saved PNG, SVG, or PDF contains only the checked series.
- The saved legend contains only the checked series.
- The live preview is restored exactly after export and is not permanently altered by the export selection.
- **Select all** and **Clear all** actions are included.
- When many legend entries are available, only the series-selection area scrolls.

## Preserved export behavior

- Live Optimization preview remains square.
- Live plot output remains exact 1:1 for PNG, SVG, and PDF.
- PNG resolution remains selectable from 600 to 2400 DPI.
- Plot tools remain organized into focused popup editors.
- No GitHub Actions workflow directory is included.
