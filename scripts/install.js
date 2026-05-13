#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Qualixar
//
// npm postinstall — bootstraps the Python package agent-amplifier on a
// system that just ran `npm install -g agent-amplifier`. The npm package
// is a thin wrapper; the actual product is Python. This script gives the
// user a single-command install path:
//
//   npm install -g agent-amplifier
//
// ...is equivalent to:
//
//   pipx install agent-amplifier
//
// Skip rules (exit 0 quietly):
//   - AGENT_AMP_SKIP_POSTINSTALL=1 in env (CI / Docker / opt-out)
//   - npm dry-run mode
//
// Failure rules (warn + exit 0 so npm install does not hard-fail):
//   - Python >= 3.11 not found
//   - pipx install fails
//   The Node bin wrapper (bin/agent-amp) prints a clear runtime error if
//   the Python binary is missing, so install can be retried with
//   `npm rebuild -g agent-amplifier` after the user fixes their env.

"use strict";

const { spawnSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const PKG_VERSION = require("../package.json").version;

function log(msg) {
  process.stdout.write(`agent-amplifier: ${msg}\n`);
}

function warn(msg) {
  process.stderr.write(`agent-amplifier: ${msg}\n`);
}

function skipIfRequested() {
  if (process.env.AGENT_AMP_SKIP_POSTINSTALL === "1") {
    log("AGENT_AMP_SKIP_POSTINSTALL=1 — skipping Python bootstrap.");
    process.exit(0);
  }
  if (process.env.npm_config_dry_run === "true") {
    log("npm dry-run — skipping Python bootstrap.");
    process.exit(0);
  }
}

function findPython() {
  const candidates =
    process.platform === "win32"
      ? ["py -3.13", "py -3.12", "py -3.11", "python3", "python"]
      : ["python3.13", "python3.12", "python3.11", "python3", "python"];
  for (const candidate of candidates) {
    const [cmd, ...args] = candidate.split(" ");
    const probe = spawnSync(
      cmd,
      [...args, "-c", "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"],
      { stdio: "ignore" }
    );
    if (probe.status === 0) return { cmd, args };
  }
  return null;
}

function run(cmd, args, opts = {}) {
  return spawnSync(cmd, args, { stdio: "inherit", ...opts });
}

function pipxAvailable(python) {
  const r = spawnSync(python.cmd, [...python.args, "-m", "pipx", "--version"], { stdio: "ignore" });
  return r.status === 0;
}

function installPipx(python) {
  log("pipx not detected — installing via `pip install --user pipx` ...");
  const r = run(python.cmd, [...python.args, "-m", "pip", "install", "--user", "--quiet", "pipx"]);
  return r.status === 0;
}

function pipxInstall(python, version) {
  log(`installing agent-amplifier==${version} via pipx ...`);
  const r = run(python.cmd, [
    ...python.args,
    "-m",
    "pipx",
    "install",
    "--force",
    `agent-amplifier==${version}`,
  ]);
  return r.status === 0;
}

function ensurePath(python) {
  // Best-effort; do not gate on its exit code.
  spawnSync(python.cmd, [...python.args, "-m", "pipx", "ensurepath"], { stdio: "inherit" });
}

function main() {
  skipIfRequested();

  const python = findPython();
  if (!python) {
    warn(
      "Python 3.11+ not found on PATH. The npm wrapper is installed but `agent-amp` will not work until Python is available.\n" +
        "  Install Python 3.11 or newer from https://www.python.org/downloads/\n" +
        "  Then run:  npm rebuild -g agent-amplifier\n" +
        "  Or install the Python package directly:  pipx install agent-amplifier"
    );
    process.exit(0);
  }
  log(`using ${[python.cmd, ...python.args].join(" ")}`);

  if (!pipxAvailable(python)) {
    if (!installPipx(python)) {
      warn(
        "Failed to install pipx automatically. Try one of:\n" +
          "  pipx install agent-amplifier\n" +
          "  pip install --user agent-amplifier"
      );
      process.exit(0);
    }
  }

  if (!pipxInstall(python, PKG_VERSION)) {
    warn(
      "pipx install of agent-amplifier failed.\n" +
        "  Try manually:  pipx install agent-amplifier==" +
        PKG_VERSION
    );
    process.exit(0);
  }

  ensurePath(python);

  log("install complete. Run:  agent-amp --help");
  log(
    "If `agent-amp` is not found, your shell PATH may need a refresh — open a new terminal, or run `pipx ensurepath` and re-source your shell config."
  );
}

main();
