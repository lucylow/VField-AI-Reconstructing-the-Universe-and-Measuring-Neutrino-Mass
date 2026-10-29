# νField Observatory frontend

A no-build, dependency-free browser frontend for cosmology concept visualization and inspection of selected νField JSON artifacts. The app is deliberately separate from the Python research pipeline; it does not run the model or claim a physical cosmological prediction.

## Run locally

From the repository root:

```bash
python scripts/serve_frontend.py
```

Then open `http://127.0.0.1:8000/frontend/`.

You can also serve the repository with any static HTTP server. ES modules generally do not work when opening `index.html` directly as a `file://` URL.

## Views

- **Overview:** animated scale-factor/redshift concept, high-level project state, navigation cards, and a local rule-based model-planning panel.
- **Cosmic expansion:** comoving marker illustration driven by `a = 1 / (1 + z)`, with synchronized epoch presets, play/pause, and reduced-motion support.
- **Cosmic web:** seeded procedural 2D network with nodes, filaments, particle flow, density adjustment, and labels.
- **CMB sky:** seeded procedural all-sky anisotropy texture, relative-amplitude histogram, graticule, contrast control, and scan animation. This is not observational CMB data.
- **Gravitational lensing:** point-mass lens equation sketch with unlensed source markers, apparent image splitting, adjustable schematic Einstein-angle scale, and animated ray sweep.
- **Power spectrum:** responsive SVG chart with logarithmic styling and editable illustrative amplitude, tilt, and small-scale damping.
- **Parameter space:** pairwise synthetic scatter plot and a renderer for imported median absolute-error metrics.
- **Run diagnostics:** coverage chart, split-overlap metadata, scientific caveats, and provenance displays for imported reports.

## Navigation and accessibility

- View routes are shareable as `#view=...` URLs and participate in browser back/forward history.
- The mobile sidebar has a backdrop, is removed from keyboard navigation while closed, and closes from outside click or Escape; the active view is announced with `aria-current`.
- Modal dialogs trap keyboard focus, close with Escape, and restore focus to the control that opened them.
- Reduced-motion preferences pause every canvas animation and update live if the operating-system setting changes. The global motion button remains available as an explicit override.

## Failure handling

- Invalid local file selections and malformed JSON return per-file errors; non-finite numbers and duplicate recognized reports are rejected safely.
- SVG export reports failure instead of throwing when browser download APIs are unavailable and releases temporary object URLs after each attempt.
- View navigation remains usable if the browser blocks History API updates; only the bookmark/back-button behavior is unavailable in that case.

## Importing results

Use **Import results** to choose JSON sidecars/reports, including:

- `dataset.json`
- `dataset.manifest.json`
- `benchmark_report.json`

Files are read with the browser File API and remain client-side. The app does not make a network request with file contents. `.npz` arrays are not decoded. A JSON file with unknown structure is shown as an unclassified artifact but does not affect scientific charts. Imports reject invalid file objects, files over 20 MB, excessive nesting, and non-finite numbers; when multiple files of the same recognized type are selected, the first is kept and the duplicate is reported rather than silently replacing it.

## Scientific and data boundaries

- Expansion and cosmic-web animations are explanatory schematics, not numerical solutions of the Friedmann equations or an N-body/particle-mesh simulation.
- The lensing view uses the point-mass lens equation in arbitrary screen-space units. It does not calculate a physical mass, angular-diameter distance, or realistic observed lens system.
- The power spectrum is an arbitrary function for exploring responsive chart interactions. It is not a CLASS/CAMB result or a neutrino transfer function.
- Scatter points are generated synthetic UI values; they are never posterior samples.
- The overview model planner uses transparent rules over the declared simulation count, parameter count, field availability, and compute budget. It is not a trained AI, performance predictor, or scientific approval.
- Imported JSON data is shown as provided. The frontend does not verify model assumptions, calibration, physical validity, or file provenance hashes.
- Benchmark metric arrays are rendered only from the relevant report fields when present. A displayed metric is not a claim of scientific validity.

## Architecture

- `index.html`: semantic app structure, view sections, modal dialogs, and accessible labels.
- `styles.css`: responsive design tokens, layout, UI states, and reduced-motion styling.
- `src/main.js`: view orchestration, import workflow, and state-to-view rendering.
- `src/core/`: tiny state store, concept math helpers, and DOM/accessibility utilities.
- `src/components/`: navigation, modal behavior, and dashboard advisor rendering.
- `src/data/`: explicit synthetic examples and bounded local JSON parsing.
- `src/visualizations/`: canvas animations and SVG chart generators.

No external JavaScript packages, analytics, fonts, or third-party requests are required.
