# PyPI Upload Fix - v2.3.0

## Issue
PyPI upload was failing with the error:
```
ERROR: InvalidDistribution: Invalid distribution metadata: unrecognized or malformed field 'license-file'
```

## Root Cause
Setuptools was automatically detecting the `LICENSE` file and adding a deprecated `License-File` metadata field to the package distribution. This field was removed in newer versions of the metadata specification and is no longer accepted by PyPI.

## Solution
Added `license-files = []` configuration to `pyproject.toml` to prevent setuptools from automatically including license file metadata.

### Changes Made

**File: `pyproject.toml`**
```toml
[tool.setuptools]
include-package-data = true
license-files = []  # ← Added this line to prevent deprecated License-File field
```

This tells setuptools to not automatically add license file references to the package metadata, while still including the LICENSE file in the distribution through MANIFEST.in.

## Build Process

1. **Clean old builds:**
   ```bash
   rm -rf dist/* build/* *.egg-info
   ```

2. **Rebuild package:**
   ```bash
   python3 -m build
   ```

3. **Verify metadata:**
   ```bash
   tar -xzf dist/fluxgraph-2.3.0.tar.gz -C /tmp
   cat /tmp/fluxgraph-2.3.0/PKG-INFO | grep -i "license"
   ```

   **Before fix:**
   - Had `License-File: LICENSE` (deprecated field)
   - Had `Dynamic: license-file` (incorrect)

   **After fix:**
   - Only has `License: MIT` (correct)
   - Only has `Classifier: License :: OSI Approved :: MIT License` (correct)

4. **Upload to PyPI:**
   ```bash
   ~/.local/bin/twine upload dist/*
   ```

## Notes
- The LICENSE file is still included in the source distribution via MANIFEST.in
- The license is still properly declared in the metadata through the `license` field in pyproject.toml
- This fix is forward-compatible with modern packaging standards

## Date
January 3, 2026

## Version
v2.3.0
