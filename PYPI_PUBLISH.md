# Publishing FluxGraph v2.2.0 to PyPI

## ✅ Pre-Publish Checklist

- [x] Version bumped to 2.2.0 in:
  - `pyproject.toml`
  - `setup.py`
- [x] CHANGELOG.md updated
- [x] README.md updated
- [x] All tests passing
- [x] Package built successfully
- [x] Code committed and pushed to GitHub

## 📦 Build Status

Package built successfully:
- ✅ `dist/fluxgraph-2.2.0.tar.gz` (source distribution)
- ✅ `dist/fluxgraph-2.2.0-py3-none-any.whl` (wheel)

## 🚀 Publishing to PyPI

### Option 1: Test PyPI (Recommended First)

```bash
# Upload to Test PyPI
~/.local/bin/twine upload --repository testpypi dist/fluxgraph-2.2.0*

# Test installation
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fluxgraph==2.2.0

# Verify
flux version
python -c "from fluxgraph.core.app import FluxApp; print('✅ Import successful')"
```

### Option 2: Production PyPI

```bash
# Upload to PyPI (PRODUCTION)
~/.local/bin/twine upload dist/fluxgraph-2.2.0*

# You will be prompted for:
# Username: __token__
# Password: <your-pypi-api-token>
```

**Note:** Make sure you have PyPI credentials set up:
- Generate API token at https://pypi.org/manage/account/token/
- Or use credentials in `~/.pypirc`

### Option 3: Using ~/.pypirc

Create `~/.pypirc` file:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

[testpypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

Then simply:
```bash
~/.local/bin/twine upload dist/fluxgraph-2.2.0*
```

## 🧪 Post-Publish Verification

After publishing, verify the release:

```bash
# Create fresh venv
python3 -m venv test_env
source test_env/bin/activate

# Install from PyPI
pip install fluxgraph[all]

# Test CLI
flux version
flux validate

# Test basic functionality
python -c "
from fluxgraph.core.app import FluxApp
app = FluxApp()
print('✅ FluxGraph v2.2.0 working!')
"

# Test new features
python -c "
from fluxgraph.core.webhooks import WebhookManager
from fluxgraph.middleware.rate_limiter import RateLimiter
from fluxgraph.core.plugins import PluginManager
print('✅ All new features importable!')
"
```

## 📊 Release Checklist

After successful PyPI publish:

- [ ] Create GitHub Release
  - Tag: `v2.2.0`
  - Title: `v2.2.0: Webhooks, Rate Limiting, Plugins & Enhanced CLI`
  - Description: Copy from CHANGELOG.md
  - Attach build artifacts

- [ ] Announce on:
  - [ ] Discord community
  - [ ] GitHub Discussions
  - [ ] Twitter/X
  - [ ] LinkedIn

- [ ] Update documentation:
  - [ ] ReadTheDocs (if needed)
  - [ ] Update examples with new features
  - [ ] Add tutorials for webhooks, rate limiting, plugins

## 🎉 What's New in v2.2.0

Highlight these features in announcements:

### 🔔 Webhooks System
Production-grade event notifications with HMAC signatures and automatic retries.

### 🚦 Rate Limiting
Protect your APIs with flexible per-user or per-IP rate limiting.

### 🔌 Plugin System
Extend FluxGraph without forking - first-class plugin API.

### 🛠️ Enhanced CLI
`flux init`, `flux validate`, `flux plugin` and more developer-friendly commands.

### 🔒 Security Fixes
Critical code injection vulnerability patched (CVE-2026-XXXX).

## 🐛 Known Issues

None reported for v2.2.0.

## 📞 Support

If issues arise during publishing:
- Check PyPI status: https://status.python.org
- Verify credentials
- Check package size limits
- Review twine logs

## 📝 Notes

- Package size: ~XXX KB (source) + ~XXX KB (wheel)
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- License: MIT
- Dependencies: See `requirements.txt` and `pyproject.toml`

---

**Ready to publish!** 🚀

Run: `~/.local/bin/twine upload dist/fluxgraph-2.2.0*`
