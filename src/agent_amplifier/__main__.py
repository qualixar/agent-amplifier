# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""``python -m agent_amplifier`` entry point — delegates to :func:`cli.main`."""

from __future__ import annotations

import sys

from agent_amplifier.cli import main

if __name__ == "__main__":
    sys.exit(main())
