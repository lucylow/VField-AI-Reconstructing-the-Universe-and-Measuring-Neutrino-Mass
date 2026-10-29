import test from "node:test";
import assert from "node:assert/strict";
import { exportSvg } from "../src/visualizations/svg-charts.js";

test("SVG export reports an unavailable chart without throwing", () => {
  assert.equal(exportSvg(null), false);
  assert.equal(exportSvg({ querySelector: () => null }), false);
});

test("SVG export attaches a download link and releases its object URL", () => {
  const originals = {
    URL: globalThis.URL,
    document: globalThis.document,
    window: globalThis.window,
    XMLSerializer: globalThis.XMLSerializer,
  };
  let appended = false;
  let clicked = false;
  let removed = false;
  let released = false;
  let cleanup;
  const svg = {
    cloneNode: () => ({ setAttribute() {} }),
  };
  globalThis.URL = {
    createObjectURL: () => "blob:test",
    revokeObjectURL: (url) => { released = url === "blob:test"; },
  };
  globalThis.XMLSerializer = class { serializeToString() { return "<svg/>"; } };
  globalThis.document = {
    body: { append: (anchor) => { appended = anchor.download === "safe-name.svg"; } },
    createElement: () => ({
      click: () => { clicked = true; },
      remove: () => { removed = true; },
    }),
  };
  globalThis.window = { setTimeout: (callback) => { cleanup = callback; } };
  try {
    assert.equal(exportSvg({ querySelector: () => svg }, "safe/name.svg"), true);
    assert.equal(appended, true);
    assert.equal(clicked, true);
    assert.equal(removed, true);
    cleanup();
    assert.equal(released, true);
  } finally {
    Object.assign(globalThis, originals);
  }
});

test("SVG export converts serialization failures to a false result", () => {
  const originals = {
    URL: globalThis.URL,
    document: globalThis.document,
    window: globalThis.window,
    XMLSerializer: globalThis.XMLSerializer,
  };
  globalThis.XMLSerializer = class { serializeToString() { throw new Error("blocked"); } };
  globalThis.document = { createElement() { throw new Error("should not reach download"); } };
  try {
    assert.equal(exportSvg({ querySelector: () => ({ cloneNode: () => ({ setAttribute() {} }) }) }), false);
  } finally {
    Object.assign(globalThis, originals);
  }
});
