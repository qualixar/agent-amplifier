# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2026 Qualixar
"""Agent Amplifier internal helpers.

PRIVATE MODULE — leading-underscore prefix marks every member here as
implementation detail. None of these names are re-exported from
``agent_amplifier.__init__`` and they are NOT part of the public API.

Stability contract:
    * Members may change shape, signature, or disappear in any minor release.
    * External consumers who import from ``agent_amplifier._internal.*``
      do so at their own risk.
"""
