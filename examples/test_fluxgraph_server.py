# _test_server.py (FIXED VERSION)
from fluxgraph import FluxApp
from fastapi import Body

app = FluxApp(
    title="FluxGraph Test API",
    version="2.0.0",
    enable_streaming=True,
    enable_sessions=True,
    enable_security=False,
    enable_orchestration=True,
    enable_mcp=True,
    enable_versioning=True,
    enable_templates=True,
    enable_analytics=True,
    auto_init_rag=False,
    log_level="WARNING"
)

# CRITICAL FIX: All agents must accept **kwargs to handle internal orchestrator parameters
@app.agent()
async def streaming_test_agent(message: str, **kwargs) -> dict:
    """Streaming test agent - accepts **kwargs for orchestrator compatibility."""
    return {"message": message, "streamable": True, "status": "success"}

@app.agent()
async def session_test_agent(message: str, session_id: str = "test", **kwargs) -> dict:
    """Session test agent - accepts **kwargs for orchestrator compatibility."""
    if app.session_manager:
        app.session_manager.add_message(session_id, "user", message)
        history = app.session_manager.get_messages(session_id)
        return {
            "message": message,
            "session_id": session_id,
            "history_count": len(history),
            "status": "success"
        }
    return {"error": "Session manager not available", "status": "error"}

@app.agent()
async def coordinator_agent(task: str, **kwargs) -> dict:
    """Coordinator agent - demonstrates agent-to-agent communication."""
    call_agent = kwargs.get('call_agent')
    if call_agent:
        result = await call_agent("worker_agent", data=task)
        return {"task": task, "coordinated": True, "worker_result": result}
    return {"task": task, "coordinated": False}

@app.agent()
async def worker_agent(data: str, **kwargs) -> dict:
    """Worker agent."""
    return {"data": data, "processed": True, "status": "success"}

# Add test endpoints AFTER agent registration
@app.api.get("/test/health")
async def test_health():
    """Health check for test suite."""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": "2025-10-05T03:00:00Z",
        "features": {
            "streaming": app.stream_manager is not None,
            "sessions": app.session_manager is not None,
            "security": app.audit_logger is not None,
            "orchestration": app.handoff_protocol is not None,
            "mcp": app.mcp_server is not None,
            "versioning": app.version_manager is not None,
            "templates": app.template_marketplace is not None,
            "analytics": app.performance_monitor is not None
        }
    }

# Phase 2: Security test endpoints
@app.api.post("/test/security/pii")
async def test_pii(text: str = Body(..., embed=True)):
    """Test PII detection."""
    if not app.pii_detector:
        return {"available": False}
    detections = app.pii_detector.detect(text)
    redacted, found = app.pii_detector.redact(text)
    return {
        "available": True,
        "original": text,
        "redacted": redacted,
        "pii_count": len(found)
    }

@app.api.post("/test/security/injection")
async def test_injection(prompt: str = Body(..., embed=True)):
    """Test prompt injection detection."""
    if not app.prompt_shield:
        return {"available": False}
    is_safe, detections = app.prompt_shield.is_safe(prompt)
    return {
        "available": True,
        "prompt": prompt,
        "is_safe": is_safe
    }

@app.api.get("/test/security/audit")
async def test_audit():
    """Test audit logging."""
    return {
        "available": app.audit_logger is not None,
        "can_verify": hasattr(app.audit_logger, 'verify_integrity') if app.audit_logger else False
    }

@app.api.post("/test/security/rbac")
async def test_rbac():
    """Test RBAC."""
    return {
        "available": app.rbac_manager is not None,
        "has_jwt": hasattr(app.rbac_manager, 'generate_jwt_token') if app.rbac_manager else False
    }

# Phase 3: Orchestration test endpoints
@app.api.get("/test/orchestration/handoff")
async def test_handoff():
    """Test handoff protocol."""
    return {"available": app.handoff_protocol is not None}

@app.api.get("/test/orchestration/hitl")
async def test_hitl():
    """Test HITL manager."""
    return {"available": app.hitl_manager is not None}

@app.api.get("/test/orchestration/batch")
async def test_batch():
    """Test batch processor."""
    return {"available": app.batch_processor is not None}

@app.api.get("/test/orchestration/adherence")
async def test_adherence():
    """Test task adherence."""
    return {"available": app.task_adherence is not None}

# Phase 4: Ecosystem test endpoints
@app.api.get("/test/ecosystem/mcp")
async def test_mcp():
    """Test MCP server."""
    return {"available": app.mcp_server is not None}

@app.api.get("/test/ecosystem/versioning")
async def test_versioning():
    """Test version manager."""
    return {"available": app.version_manager is not None}

@app.api.get("/test/ecosystem/templates")
async def test_templates():
    """Test template marketplace."""
    return {"available": app.template_marketplace is not None}

@app.api.get("/test/ecosystem/multimodal")
async def test_multimodal():
    """Test multimodal processor."""
    return {"available": app.multimodal_processor is not None}

if __name__ == "__main__":
    print("=" * 80)
    print("FluxGraph Test Server")
    print("=" * 80)
    print("Health: http://localhost:8000/test/health")
    print("Docs:   http://localhost:8000/docs")
    print("=" * 80)
    app.run(port=8000)
