// agent-amplifier on npm is a metadata-only name reservation.
//
// The actual product is a Python package distributed on PyPI.
// Install with:
//
//     pip install agent-amplifier
//
// There is no JavaScript / Node API at this time.  This module exists
// only so the npm name `agent-amplifier` cannot be squatted while we
// finish the Python beta.  Stage 10 CODEX-050 — be honest about scope.
//
// If a JavaScript bridge ships in the future, this module's exports
// will change.  Until then, treat any output here as documentation.

module.exports = {
  metadataOnly: true,
  pypiPackage: "agent-amplifier",
  installCommand: "pip install agent-amplifier",
  homepage: "https://github.com/qualixar/agent-amplifier",
  version: "0.0.1",
  author: "Qualixar",
  license: "Apache-2.0",
};
