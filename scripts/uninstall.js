#!/usr/bin/env node
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2026 Qualixar
//
// npm preuninstall — best-effort cleanup of the pipx-installed Python
// package when the npm wrapper is removed. Never fails the uninstall:
// missing pipx, missing Python, or already-uninstalled all exit 0.

"use strict";

const { spawnSync } = require("child_process");

if (process.env.AGENT_AMP_SKIP_POSTINSTALL === "1") {
  process.exit(0);
}

const candidates =
  process.platform === "win32"
    ? ["py -3.13", "py -3.12", "py -3.11", "python3", "python"]
    : ["python3.13", "python3.12", "python3.11", "python3", "python"];

for (const candidate of candidates) {
  const [cmd, ...args] = candidate.split(" ");
  const probe = spawnSync(
    cmd,
    [...args, "-m", "pipx", "--version"],
    { stdio: "ignore" }
  );
  if (probe.status === 0) {
    spawnSync(cmd, [...args, "-m", "pipx", "uninstall", "agent-amplifier"], {
      stdio: "inherit",
    });
    break;
  }
}

process.exit(0);
