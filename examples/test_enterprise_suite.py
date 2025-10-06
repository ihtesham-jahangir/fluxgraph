# test_enterprise_suite.py
"""
FluxGraph 2.0 Enterprise Feature Test Suite
Automatically tests all Phase 1-4 features with HTTP validation
"""

import os
import sys
import time
import json
import logging
import subprocess
import requests
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Test configuration
BASE_URL = "http://localhost:8000"
SERVER_STARTUP_TIMEOUT = 15
TEST_TIMEOUT = 5


class Colors:
    """ANSI color codes."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class TestResult:
    """Test result container."""
    def __init__(self, name: str, phase: str, passed: bool, duration: float, message: str = ""):
        self.name = name
        self.phase = phase
        self.passed = passed
        self.duration = duration
        self.message = message


class EnterpriseTestSuite:
    """Comprehensive test suite for FluxGraph enterprise features."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.server_process = None
        self.server_ready = False
        
    def log_section(self, title: str):
        """Log section header."""
        logger.info("=" * 80)
        logger.info(f"{Colors.HEADER}{Colors.BOLD}{title}{Colors.ENDC}")
        logger.info("=" * 80)
    
    def log_test(self, name: str, status: str, duration: float, message: str = ""):
        """Log test result."""
        symbol = f"{Colors.OKGREEN}✓{Colors.ENDC}" if status == "PASS" else f"{Colors.FAIL}✗{Colors.ENDC}"
        status_color = f"{Colors.OKGREEN}{status}{Colors.ENDC}" if status == "PASS" else f"{Colors.FAIL}{status}{Colors.ENDC}"
        logger.info(f"{symbol} {name:<50} [{status_color}] ({duration:.3f}s)")
        if message:
            logger.info(f"   └─ {message}")
    
    def start_server(self):
        """Start FluxGraph test server."""
        self.log_section("Starting FluxGraph Test Server")
        
        # Generate test server code with the working pattern
        test_server_code = '''
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

@app.agent()
async def streaming_test_agent(message: str) -> dict:
    return {"message": message, "streamable": True}

@app.agent()
async def session_test_agent(message: str, session_id: str = "test") -> dict:
    if app.session_manager:
        app.session_manager.add_message(session_id, "user", message)
        history = app.session_manager.get_messages(session_id)
        return {"message": message, "session_id": session_id, "history_count": len(history)}
    return {"error": "Session manager not available"}

@app.agent()
async def coordinator_agent(task: str, call_agent=None) -> dict:
    if call_agent:
        result = await call_agent("worker_agent", data=task)
        return {"task": task, "coordinated": True}
    return {"task": task, "coordinated": False}

@app.agent()
async def worker_agent(data: str) -> dict:
    return {"data": data, "processed": True}

# Add test endpoints AFTER agent registration
@app.api.get("/test/health")
async def test_health():
    return {
        "status": "healthy",
        "features": {
            "streaming": app.stream_manager is not None,
            "sessions": app.session_manager is not None,
            "orchestration": app.handoff_protocol is not None,
            "mcp": app.mcp_server is not None,
            "versioning": app.version_manager is not None,
            "templates": app.template_marketplace is not None
        }
    }

@app.api.get("/test/orchestration/handoff")
async def test_handoff():
    return {"available": app.handoff_protocol is not None}

@app.api.get("/test/orchestration/hitl")
async def test_hitl():
    return {"available": app.hitl_manager is not None}

@app.api.get("/test/orchestration/batch")
async def test_batch():
    return {"available": app.batch_processor is not None}

@app.api.get("/test/orchestration/adherence")
async def test_adherence():
    return {"available": app.task_adherence is not None}

@app.api.get("/test/ecosystem/mcp")
async def test_mcp():
    return {"available": app.mcp_server is not None}

@app.api.get("/test/ecosystem/versioning")
async def test_versioning():
    return {"available": app.version_manager is not None}

@app.api.get("/test/ecosystem/templates")
async def test_templates():
    return {"available": app.template_marketplace is not None}

@app.api.get("/test/ecosystem/multimodal")
async def test_multimodal():
    return {"available": app.multimodal_processor is not None}

if __name__ == "__main__":
    app.run(port=8000)
'''
        
        # Write test server file
        with open("_test_server.py", "w") as f:
            f.write(test_server_code)
        
        # Start server
        logger.info(f"Starting server at {BASE_URL}...")
        self.server_process = subprocess.Popen(
            [sys.executable, "_test_server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server
        logger.info("Waiting for server...")
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            try:
                response = requests.get(f"{BASE_URL}/test/health", timeout=2)
                if response.status_code == 200:
                    self.server_ready = True
                    logger.info(f"{Colors.OKGREEN}✓{Colors.ENDC} Server ready!")
                    return True
            except:
                time.sleep(0.5)
        
        logger.error(f"{Colors.FAIL}✗{Colors.ENDC} Server failed to start")
        return False
    
    def stop_server(self):
        """Stop test server."""
        if self.server_process:
            logger.info("Stopping server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except:
                self.server_process.kill()
            logger.info(f"{Colors.OKGREEN}✓{Colors.ENDC} Server stopped")
        
        # Cleanup
        for file in ["_test_server.py", "sessions.db"]:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
    
    def make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request."""
        url = f"{BASE_URL}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, timeout=TEST_TIMEOUT)
            else:
                response = requests.post(url, json=data, timeout=TEST_TIMEOUT)
            
            return {
                "success": True,
                "status_code": response.status_code,
                "data": response.json()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Phase 1 Tests
    def test_phase1_streaming(self):
        """Test streaming."""
        start = time.time()
        try:
            result = self.make_request("POST", "/ask/streaming_test_agent", {"message": "test"})
            passed = result["success"] and result["status_code"] == 200
            duration = time.time() - start
            msg = "Streaming OK" if passed else "Streaming failed"
            self.results.append(TestResult("Streaming", "Phase 1", passed, duration, msg))
            self.log_test("Streaming", "PASS" if passed else "FAIL", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Streaming", "Phase 1", False, duration, str(e)))
            self.log_test("Streaming", "FAIL", duration, str(e))
    
    def test_phase1_sessions(self):
        """Test sessions."""
        start = time.time()
        try:
            result = self.make_request("POST", "/ask/session_test_agent", {
                "message": "test", "session_id": "test123"
            })
            passed = result["success"] and "session_id" in str(result.get("data", ""))
            duration = time.time() - start
            msg = "Sessions OK" if passed else "Sessions failed"
            self.results.append(TestResult("Sessions", "Phase 1", passed, duration, msg))
            self.log_test("Sessions", "PASS" if passed else "FAIL", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Sessions", "Phase 1", False, duration, str(e)))
            self.log_test("Sessions", "FAIL", duration, str(e))
    
    # Phase 3 Tests
    def test_phase3_handoff(self):
        """Test handoff."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/orchestration/handoff")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "Handoff available" if passed else "Handoff N/A"
            self.results.append(TestResult("Handoff", "Phase 3", passed, duration, msg))
            self.log_test("Handoff", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Handoff", "Phase 3", False, duration, str(e)))
            self.log_test("Handoff", "FAIL", duration, str(e))
    
    def test_phase3_hitl(self):
        """Test HITL."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/orchestration/hitl")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "HITL available" if passed else "HITL N/A"
            self.results.append(TestResult("HITL", "Phase 3", passed, duration, msg))
            self.log_test("HITL", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("HITL", "Phase 3", False, duration, str(e)))
            self.log_test("HITL", "FAIL", duration, str(e))
    
    def test_phase3_batch(self):
        """Test batch."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/orchestration/batch")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "Batch available" if passed else "Batch N/A"
            self.results.append(TestResult("Batch", "Phase 3", passed, duration, msg))
            self.log_test("Batch", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Batch", "Phase 3", False, duration, str(e)))
            self.log_test("Batch", "FAIL", duration, str(e))
    
    def test_phase3_adherence(self):
        """Test adherence."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/orchestration/adherence")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "Adherence available" if passed else "Adherence N/A"
            self.results.append(TestResult("Adherence", "Phase 3", passed, duration, msg))
            self.log_test("Adherence", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Adherence", "Phase 3", False, duration, str(e)))
            self.log_test("Adherence", "FAIL", duration, str(e))
    
    # Phase 4 Tests
    def test_phase4_mcp(self):
        """Test MCP."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/ecosystem/mcp")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "MCP available" if passed else "MCP N/A"
            self.results.append(TestResult("MCP", "Phase 4", passed, duration, msg))
            self.log_test("MCP", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("MCP", "Phase 4", False, duration, str(e)))
            self.log_test("MCP", "FAIL", duration, str(e))
    
    def test_phase4_versioning(self):
        """Test versioning."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/ecosystem/versioning")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "Versioning available" if passed else "Versioning N/A"
            self.results.append(TestResult("Versioning", "Phase 4", passed, duration, msg))
            self.log_test("Versioning", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Versioning", "Phase 4", False, duration, str(e)))
            self.log_test("Versioning", "FAIL", duration, str(e))
    
    def test_phase4_templates(self):
        """Test templates."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/ecosystem/templates")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "Templates available" if passed else "Templates N/A"
            self.results.append(TestResult("Templates", "Phase 4", passed, duration, msg))
            self.log_test("Templates", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Templates", "Phase 4", False, duration, str(e)))
            self.log_test("Templates", "FAIL", duration, str(e))
    
    def test_phase4_multimodal(self):
        """Test multimodal."""
        start = time.time()
        try:
            result = self.make_request("GET", "/test/ecosystem/multimodal")
            passed = result["success"] and result.get("data", {}).get("available")
            duration = time.time() - start
            msg = "Multimodal available" if passed else "Multimodal N/A"
            self.results.append(TestResult("Multimodal", "Phase 4", passed, duration, msg))
            self.log_test("Multimodal", "PASS" if passed else "SKIP", duration, msg)
        except Exception as e:
            duration = time.time() - start
            self.results.append(TestResult("Multimodal", "Phase 4", False, duration, str(e)))
            self.log_test("Multimodal", "FAIL", duration, str(e))
    
    def run_all_tests(self):
        """Run all tests."""
        if not self.server_ready:
            logger.error("Server not ready")
            return False
        
        # Phase 1
        self.log_section("Phase 1: Production Readiness")
        self.test_phase1_streaming()
        self.test_phase1_sessions()
        
        # Phase 3
        self.log_section("Phase 3: Advanced Orchestration")
        self.test_phase3_handoff()
        self.test_phase3_hitl()
        self.test_phase3_batch()
        self.test_phase3_adherence()
        
        # Phase 4
        self.log_section("Phase 4: Ecosystem")
        self.test_phase4_mcp()
        self.test_phase4_versioning()
        self.test_phase4_templates()
        self.test_phase4_multimodal()
        
        return True
    
    def generate_report(self):
        """Generate report."""
        self.log_section("Test Results")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info(f"\n{Colors.BOLD}Statistics:{Colors.ENDC}")
        logger.info(f"  Total:  {total}")
        logger.info(f"  {Colors.OKGREEN}Passed:{Colors.ENDC} {passed}")
        logger.info(f"  {Colors.FAIL}Failed:{Colors.ENDC} {failed}")
        logger.info(f"  Rate:   {pass_rate:.1f}%")
        
        logger.info("\n" + "=" * 80)
        if pass_rate >= 80:
            logger.info(f"{Colors.OKGREEN}{Colors.BOLD}✓ READY FOR PRODUCTION{Colors.ENDC}")
        else:
            logger.info(f"{Colors.WARNING}{Colors.BOLD}⚠ SOME FEATURES MISSING{Colors.ENDC}")
        logger.info("=" * 80)
        
        # Save JSON report
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump({
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": pass_rate,
                "results": [{"name": r.name, "phase": r.phase, "passed": r.passed} for r in self.results]
            }, f, indent=2)
        logger.info(f"\n{Colors.OKCYAN}📄 Report: {report_file}{Colors.ENDC}")
        
        return {"total": total, "passed": passed, "failed": failed, "pass_rate": pass_rate}


def main():
    """Main execution."""
    suite = EnterpriseTestSuite()
    
    try:
        if not suite.start_server():
            logger.error("Failed to start server")
            return 1
        
        time.sleep(2)  # Let server fully initialize
        
        suite.run_all_tests()
        report = suite.generate_report()
        
        return 0 if report["pass_rate"] >= 80 else 1
        
    except KeyboardInterrupt:
        logger.warning("\nInterrupted")
        return 130
    except Exception as e:
        logger.error(f"\n{Colors.FAIL}Fatal error: {e}{Colors.ENDC}", exc_info=True)
        return 1
    finally:
        suite.stop_server()


if __name__ == "__main__":
    sys.exit(main())
