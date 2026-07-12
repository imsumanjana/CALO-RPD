# CALO-RPD Studio v1.0.4

## Live Optimization plotting

- The live convergence preview is rendered on an exact square (1:1) Matplotlib canvas.
- The Live Optimization workspace uses vertical scrolling when required, preserving the square plot rather than compressing it.
- Export from the live plot is forced to a square physical figure for PNG, SVG, and PDF.
- Square export locks width and height together.
- Tight bounding-box cropping is disabled for square export because cropping can change the final pixel dimensions away from 1:1.

## High-resolution PNG export

- PNG DPI is selectable from 600 through 2400 DPI.
- The default PNG export resolution is 600 DPI.
- SVG and PDF are treated as vector formats; the DPI selector is disabled when those formats are selected.

## Repository automation

- GitHub Actions workflow files have been removed.
- The guided workflow remains part of the CALO-RPD Studio desktop application only.
