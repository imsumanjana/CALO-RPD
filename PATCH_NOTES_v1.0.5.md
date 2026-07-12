# CALO-RPD Studio v1.0.5

## Organized scientific plot tools

The scientific plot controls are reorganized into a compact icon-based tool strip. The plot surface no longer displays every text, axis, series, export, and style control simultaneously.

Four focused tools are available above every scientific plot:

- **Text & labels** — font family, independent font sizes, bold/italic formatting, plot title, axis labels, tick labels, legend typography, legend labels, and annotations.
- **Plot appearance** — axis scales, explicit axis limits, major/minor grids, axis line width, line style, line width, marker type, marker size, and series visibility.
- **Export figure** — PNG, SVG, and PDF export, PNG resolution from 600 to 2400 DPI, physical figure size, background transparency, and square-export handling.
- **Style profiles** — save, load, reset, and apply the current style to compatible plots.

Each tool opens a focused popup and closes when the user clicks elsewhere. The complete publication-formatting capability remains available without occupying a large permanent toolbar area.

## Preserved behavior

- Live Optimization preview remains exactly square.
- Live figure export remains exact 1:1 for PNG, SVG, and PDF.
- PNG export remains selectable from 600 to 2400 DPI.
- Plot text and style changes continue to update the visible Matplotlib figure immediately.
- Raw scientific data remain separate from presentation styling.
- The repository contains no GitHub Actions workflow configuration.
