// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Qualixar
//
// agent-amplifier — npm wrapper for the Python package distributed on
// PyPI. Installing via npm runs a postinstall script (scripts/install.js)
// that bootstraps the Python package through pipx, and exposes the
// `agent-amp` CLI via bin/agent-amp. The Python product is the source of
// truth — this module just describes the bridge.
//
// Programmatic Node API:
//
//   const aa = require("agent-amplifier");
//   aa.run(["doctor", "--json"]);   // spawns the pipx-installed agent-amp
//
// For the actual amplification logic, install the Python package and
// import it from Python: `from agent_amplifier import AgentAmplifier`.

"use strict";

const { spawnSync, spawn } = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const PKG = require("./package.json");

function isExecutable(p) {
  try {
    fs.accessSync(p, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function resolveBinary() {
  const home = os.homedir();
  const candidates = [
    path.join(home, ".local", "bin", "agent-amp"),
    path.join(home, ".local", "pipx", "venvs", "agent-amplifier", "bin", "agent-amp"),
    path.join(home, "Library", "Application Support", "pipx", "venvs", "agent-amplifier", "bin", "agent-amp"),
    path.join(home, ".local", "bin", "agent-amp.exe"),
    path.join(home, "AppData", "Local", "pipx", "pipx", "venvs", "agent-amplifier", "Scripts", "agent-amp.exe"),
  ];
  for (const c of candidates) {
    if (isExecutable(c)) return c;
  }
  const lookup = process.platform === "win32" ? "where" : "which";
  const r = spawnSync(lookup, ["agent-amp"], { encoding: "utf8" });
  if (r.status === 0) {
    const first = (r.stdout || "").split(/\r?\n/).find((line) => line && line.trim());
    if (first && isExecutable(first.trim())) return first.trim();
  }
  return null;
}

function run(argv = [], options = {}) {
  const target = resolveBinary();
  if (!target) {
    throw new Error(
      "agent-amp: Python install not found. Run `pipx install agent-amplifier` or `npm rebuild -g agent-amplifier`."
    );
  }
  return spawnSync(target, argv, { stdio: "inherit", ...options });
}

function runAsync(argv = [], options = {}) {
  const target = resolveBinary();
  if (!target) {
    return Promise.reject(
      new Error(
        "agent-amp: Python install not found. Run `pipx install agent-amplifier` or `npm rebuild -g agent-amplifier`."
      )
    );
  }
  return new Promise((resolve, reject) => {
    const child = spawn(target, argv, { stdio: "inherit", ...options });
    child.on("error", reject);
    child.on("exit", (code, signal) => resolve({ code, signal }));
  });
}

module.exports = {
  /** Package version — kept in lockstep with the PyPI release. */
  version: PKG.version,
  /** Canonical name on PyPI. */
  pypiPackage: "agent-amplifier",
  /** Resolve the on-disk path of the installed Python CLI, or null. */
  resolveBinary,
  /** Synchronously invoke the installed agent-amp CLI. */
  run,
  /** Asynchronously invoke the installed agent-amp CLI. */
  runAsync,
  /** Project homepage. */
  homepage: "https://github.com/qualixar/agent-amplifier",
  /** SPDX license identifier. */
  license: "AGPL-3.0-or-later",
};
