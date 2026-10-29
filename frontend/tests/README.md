# Frontend validation checklist

Run these manually in a browser while serving the repository with `python scripts/serve_frontend.py`:

1. Confirm each navigation item opens one view; the active view and breadcrumb match.
2. Drag the redshift sliders in both overview and detail views. Both controls and scale-factor readouts should stay synchronized.
3. Use each epoch preset in both expansion views; confirm the active preset, both sliders, and scale-factor readouts stay synchronized.
4. Change the dashboard model-planner inputs; verify its recommendation and ratio update, and invalid values show an actionable prompt.
5. Use expansion Play / Pause, reset, the global motion control, and system reduced-motion settings.
6. Regenerate the cosmic-web network; change density and labels; pause/resume particle flow.
7. Move all power-spectrum sliders; confirm the SVG curve redraws and Export SVG downloads a local file.
8. Change each parameter axis; confirm the scatter plot and tooltips update.
9. Import `examples/sample_output/dataset.json` and `examples/sample_output/dataset.manifest.json`.
10. Import a benchmark report containing `coverage_diagnostic`, `median_absolute_error`, and split IDs.
11. Try malformed JSON, an unsupported file extension, a large file, and an unknown JSON object. The page should show a message rather than crash.
12. Test at 320px, tablet width, and desktop width; verify labels, dialogs, focus styles, and reduced-motion handling.
13. Open each dialog by keyboard, tab forward/backward through its controls, close with Escape, and confirm focus returns to the opener.
14. At mobile width, verify the closed sidebar cannot receive keyboard focus; open it with the menu button, navigate by keyboard, and check Escape/scrim close behavior.
15. Use browser developer tools to confirm there are no uncaught exceptions and no requests for imported file contents.

This manual checklist is not a substitute for automated browser integration tests; this frontend currently has no browser-test runner dependency.
