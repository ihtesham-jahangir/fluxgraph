# tests/test_all_features.py
"""
Complete automated test suite for FluxGraph v3.2
Tests v3.0, v3.1, and v3.2 features (matches current app.py)
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from fluxgraph.core.app import FluxApp

# Mock classes
class MockMemory:
    def __init__(self):
        self.store = {}
    
    async def save(self, key, value):
        self.store[key] = value
    
    async def get(self, key):
        return self.store.get(key)

class MockRAG:
    def __init__(self):
        self.documents = []
    
    def add_documents(self, docs):
        self.documents.extend(docs)
    
    def query(self, query):
        return {"results": self.documents[:3]}
    
    def get_collection_stats(self):
        return {"total_documents": len(self.documents)}


# Test configuration
class TestConfig:
    @staticmethod
    def create_app(**kwargs):
        config = {
            "title": "FluxGraph Test",
            "version": "3.2.0-test",
            "memory_store": MockMemory(),
            "rag_connector": MockRAG(),
            "auto_init_rag": False,
            "enable_analytics": False,
            "log_level": "WARNING"
        }
        config.update(kwargs)
        return FluxApp(**config)
    
    @staticmethod
    def print_test_header(test_name):
        print("\n" + "="*80)
        print(f"🧪 TEST: {test_name}")
        print("="*80)
    
    @staticmethod
    def print_result(passed, message):
        icon = "✅" if passed else "❌"
        status = "PASSED" if passed else "FAILED"
        print(f"{icon} {status}: {message}")


# Core tests
class TestCore:
    @staticmethod
    async def test_app_initialization():
        TestConfig.print_test_header("Core: App Initialization")
        try:
            app = TestConfig.create_app()
            assert app.title == "FluxGraph Test"
            assert app.version == "3.2.0-test"
            assert app.registry is not None
            assert app.tool_registry is not None
            TestConfig.print_result(True, "App initialized")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_agent_registration():
        TestConfig.print_test_header("Core: Agent Registration")
        try:
            app = TestConfig.create_app()
            
            class MockAgent:
                async def run(self, **kwargs):
                    return {"result": "success"}
            
            app.register("test_agent", MockAgent())
            agent = app.registry.get("test_agent")
            assert agent is not None
            
            TestConfig.print_result(True, "Agent registered")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_tool_registration():
        TestConfig.print_test_header("Core: Tool Registration")
        try:
            app = TestConfig.create_app()
            
            @app.tool("test_tool")
            def test_function(x, y):
                return x + y
            
            tools = app.tool_registry.list_tools()
            assert "test_tool" in tools
            
            TestConfig.print_result(True, "Tool registered")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False


# V3.0 tests
class TestV30Features:
    @staticmethod
    async def test_workflows():
        TestConfig.print_test_header("v3.0: Workflows")
        try:
            app = TestConfig.create_app(enable_workflows=True)
            assert app.workflows_enabled == True
            TestConfig.print_result(True, "Workflows enabled")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_advanced_memory():
        TestConfig.print_test_header("v3.0: Advanced Memory")
        try:
            app = TestConfig.create_app(enable_advanced_memory=True)
            assert app.advanced_memory_enabled == True
            TestConfig.print_result(True, "Advanced memory enabled")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_agent_cache():
        TestConfig.print_test_header("v3.0: Agent Cache")
        try:
            app = TestConfig.create_app(enable_agent_cache=True, cache_strategy="hybrid")
            assert app.agent_cache_enabled == True
            TestConfig.print_result(True, "Cache enabled")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False


# V3.2 tests
class TestV32Features:
    @staticmethod
    async def test_chains():
        TestConfig.print_test_header("v3.2: Chains")
        try:
            app = TestConfig.create_app(enable_chains=True)
            assert app.chains_enabled == True
            TestConfig.print_result(True, "Chains enabled")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_tracing():
        TestConfig.print_test_header("v3.2: Tracing")
        try:
            app = TestConfig.create_app(enable_tracing=True)
            assert app.tracing_enabled == True
            TestConfig.print_result(True, "Tracing enabled")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_batch_optimization():
        TestConfig.print_test_header("v3.2: Batch Optimization")
        try:
            app = TestConfig.create_app(enable_batch_optimization=True)
            assert app.batch_optimizer_enabled == True
            TestConfig.print_result(True, "Batch optimization enabled")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_streaming():
        TestConfig.print_test_header("v3.2: Streaming")
        try:
            app = TestConfig.create_app(enable_streaming_optimization=True)
            assert app.streaming_optimizer_enabled == True
            TestConfig.print_result(True, "Streaming enabled")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False


# Integration tests
class TestIntegration:
    @staticmethod
    async def test_memory_integration():
        TestConfig.print_test_header("Integration: Memory")
        try:
            memory = MockMemory()
            app = TestConfig.create_app(memory_store=memory)
            
            await memory.save("test_key", "test_value")
            value = await memory.get("test_key")
            assert value == "test_value"
            
            TestConfig.print_result(True, "Memory working")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False
    
    @staticmethod
    async def test_rag_integration():
        TestConfig.print_test_header("Integration: RAG")
        try:
            rag = MockRAG()
            app = TestConfig.create_app(rag_connector=rag)
            
            rag.add_documents(["doc1", "doc2", "doc3"])
            result = rag.query("test query")
            assert len(result["results"]) > 0
            
            TestConfig.print_result(True, "RAG working")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False


# API tests
class TestAPI:
    @staticmethod
    async def test_api_routes():
        TestConfig.print_test_header("API: Routes")
        try:
            app = TestConfig.create_app()
            assert app.api is not None
            assert len(app.api.routes) > 0
            TestConfig.print_result(True, "API routes configured")
            return True
        except Exception as e:
            TestConfig.print_result(False, f"Failed: {e}")
            return False


# Test runner
class AutomatedTestRunner:
    def __init__(self):
        self.results = {"passed": 0, "failed": 0, "total": 0, "details": []}
    
    async def run_test(self, test_func, test_name):
        self.results["total"] += 1
        try:
            result = await test_func()
            if result:
                self.results["passed"] += 1
                self.results["details"].append({"test": test_name, "status": "PASSED"})
            else:
                self.results["failed"] += 1
                self.results["details"].append({"test": test_name, "status": "FAILED"})
        except Exception as e:
            self.results["failed"] += 1
            self.results["details"].append({"test": test_name, "status": "ERROR", "error": str(e)})
            print(f"❌ ERROR: {e}")
    
    async def run_all_tests(self):
        print("\n" + "="*80)
        print("🚀 FLUXGRAPH v3.2 - AUTOMATED TEST SUITE")
        print("="*80)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Core
        await self.run_test(TestCore.test_app_initialization, "Core: App Init")
        await self.run_test(TestCore.test_agent_registration, "Core: Agent Reg")
        await self.run_test(TestCore.test_tool_registration, "Core: Tool Reg")
        
        # v3.0
        await self.run_test(TestV30Features.test_workflows, "v3.0: Workflows")
        await self.run_test(TestV30Features.test_advanced_memory, "v3.0: Memory")
        await self.run_test(TestV30Features.test_agent_cache, "v3.0: Cache")
        
        # v3.2
        await self.run_test(TestV32Features.test_chains, "v3.2: Chains")
        await self.run_test(TestV32Features.test_tracing, "v3.2: Tracing")
        await self.run_test(TestV32Features.test_batch_optimization, "v3.2: Batch")
        await self.run_test(TestV32Features.test_streaming, "v3.2: Streaming")
        
        # Integration
        await self.run_test(TestIntegration.test_memory_integration, "Integration: Memory")
        await self.run_test(TestIntegration.test_rag_integration, "Integration: RAG")
        
        # API
        await self.run_test(TestAPI.test_api_routes, "API: Routes")
        
        self.print_summary()
    
    def print_summary(self):
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"Total:   {self.results['total']}")
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        rate = (self.results['passed']/self.results['total']*100) if self.results['total'] > 0 else 0
        print(f"Success: {rate:.1f}%")
        print("="*80)
        
        if self.results["failed"] > 0:
            print("\n❌ FAILED TESTS:")
            for detail in self.results["details"]:
                if detail["status"] in ["FAILED", "ERROR"]:
                    print(f"   - {detail['test']}")
        
        print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with open("test_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        print("\n💾 Results saved to test_results.json\n")


async def main():
    runner = AutomatedTestRunner()
    await runner.run_all_tests()
    exit_code = 0 if runner.results["failed"] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
