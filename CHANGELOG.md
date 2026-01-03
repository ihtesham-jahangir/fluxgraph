# Changelog

## [2.2.0] - 2026-01-02

### 🚀 Major Update: Enhanced Developer Experience

#### New Features
- ✅ **Webhook System** - Production-grade event notifications with HMAC signatures and auto-retries
- ✅ **Rate Limiting** - Flexible rate limiting with Redis and in-memory backends
- ✅ **Plugin System** - First-class extensibility with plugin API
- ✅ **Enhanced CLI** - Comprehensive command-line tools (init, test, validate, export, plugin management)
- ✅ **Security Fixes** - Replaced unsafe eval() with AST-based expression evaluator (CVE-2026-XXXX)

#### Security Improvements
- 🔒 Fixed CRITICAL code injection vulnerability in workflow evaluation
- 🔒 Improved error handling (replaced bare except clauses)
- 🔒 Added HMAC signature support for webhooks
- 🔒 Better dependency tracking for optional features

#### Bug Fixes
- Fixed missing `os` import in main.py
- Corrected FluxApp import paths
- Fixed variable reference errors in core.app
- Created missing __init__.py files for multimodal, security, orchestration, protocols packages
- Removed circular import risks

#### Developer Experience
- 📦 `flux init` - Create new projects with templates
- 📦 `flux validate` - Validate setup and dependencies
- 📦 `flux plugin` - Manage plugins easily
- 📦 `flux docs` - Open documentation directly
- 📝 Completely redesigned README with better examples
- 📝 Comprehensive bug fixes report

## [2.1.0] - 2025-11-16

### Stability & Performance Release
- Performance optimizations
- Bug fixes and stability improvements

## [2.0.0] - 2025-10-05

### 🎉 Major Release: Enterprise Edition

#### Phase 1: Production Readiness
- ✅ Streaming responses (SSE)
- ✅ Session management (SQLite + PostgreSQL)
- ✅ Retry logic with exponential backoff
- ✅ Output validation with Pydantic schemas

#### Phase 2: Enterprise Security
- ✅ Immutable audit logs (blockchain-style)
- ✅ PII detection (9 types)
- ✅ Prompt injection shields (7 techniques)
- ✅ RBAC + JWT authentication

#### Phase 3: Advanced Orchestration
- ✅ Agent handoff protocol (A2A)
- ✅ Human-in-the-loop workflows
- ✅ Task adherence monitoring
- ✅ Batch processing with priority queues

#### Phase 4: Ecosystem Growth
- ✅ MCP protocol support
- ✅ Agent versioning & A/B testing
- ✅ Agent template marketplace
- ✅ Multi-modal support (images + audio)

#### Unique Features
- ✅ Circuit breakers (only in FluxGraph)
- ✅ Real-time cost tracking per agent
- ✅ Smart AI-powered routing

### Breaking Changes
- Minimum Python version: 3.8
- New security features require additional dependencies
- API structure reorganized for better modularity

## [0.0.5] - 2024-XX-XX
- Initial MVP release

## [0.0.1] - 2024-XX-XX
- Initial development version
