# File: examples/analytics_example.py
"""
Example usage of FluxGraph with Performance Analytics and UI access.
"""

import asyncio
import random
from fluxgraph import FluxApp
from fluxgraph.analytics import PerformanceMonitor, MetricsCollector, AnalyticsDashboard

# Initialize the app with analytics enabled
app = FluxApp(
    title="Analytics Demo",
    description="A lightweight Python framework for building, orchestrating, and deploying Agentic AI systems.",
    version="0.1.0",
    enable_analytics=True
)

# Initialize MetricsCollector (manually for this example)
metrics_collector = MetricsCollector()

# Sample agents with **kwargs to handle framework arguments like 'tools'
@app.agent()
async def fast_agent(message: str, user_id: str = None, session_id: str = None, **kwargs) -> dict:
    """A fast-responding agent."""
    await asyncio.sleep(0.1)  # Simulate quick processing
    metrics_collector.increment('agent_executions', labels={'agent': 'fast_agent'})
    metrics_collector.observe('custom_duration', 0.1, labels={'agent': 'fast_agent'})
    return {"response": f"Quick response to: {message}", "processing_time": "100ms"}

@app.agent()
async def slow_agent(query: str, user_id: str = None, session_id: str = None, **kwargs) -> dict:
    """A slower agent that simulates complex processing."""
    await asyncio.sleep(1.5)  # Simulate slower processing
    metrics_collector.increment('agent_executions', labels={'agent': 'slow_agent'})
    metrics_collector.observe('custom_duration', 1.5, labels={'agent': 'slow_agent'})
    return {"response": f"Processed query: {query}", "processing_time": "1500ms"}

@app.agent()
async def llm_agent(prompt: str, user_id: str = None, session_id: str = None, **kwargs) -> dict:
    """An agent that simulates LLM usage."""
    await asyncio.sleep(0.8)  # Simulate LLM API call
    tokens_used = len(prompt.split()) * 1.3  # Rough estimation
    cost = tokens_used * 0.00002  # $0.00002 per token
    
    # Manually update PerformanceMonitor with token and cost info
    if app.performance_monitor:
        request_id = app.performance_monitor.start_tracking(
            agent_name='llm_agent',
            request_data=prompt,
            user_id=user_id,
            session_id=session_id
        )
        app.performance_monitor.finish_tracking(
            request_id=request_id,
            success=True,
            output_data=prompt,
            tokens_used=int(tokens_used),
            cost_usd=cost
        )
    
    metrics_collector.increment('agent_executions', labels={'agent': 'llm_agent'})
    metrics_collector.observe('custom_duration', 0.8, labels={'agent': 'llm_agent'})
    
    return {
        "response": f"AI response to: {prompt}",
        "tokens_used": int(tokens_used),
        "estimated_cost": f"${cost:.6f}"
    }

@app.agent()
async def error_prone_agent(data: str, user_id: str = None, session_id: str = None, **kwargs) -> dict:
    """An agent that sometimes fails."""
    metrics_collector.increment('agent_executions', labels={'agent': 'error_prone_agent'})
    metrics_collector.observe('custom_duration', 0.5, labels={'agent': 'error_prone_agent'})
    
    if random.random() < 0.3:  # 30% failure rate
        raise ValueError("Simulated agent failure")
    return {"response": f"Processed data: {data}"}

# Simulate agent calls to populate metrics (updated to pass only expected args)
async def simulate_agent_calls():
    """Simulate multiple agent calls to generate metrics."""
    tasks = [
        fast_agent("Hello, fast!", user_id="user1", session_id="session1"),
        slow_agent("Complex query", user_id="user1", session_id="session1"),
        llm_agent("Generate a story", user_id="user2", session_id="session2"),
        error_prone_agent("Test data", user_id="user3", session_id="session3"),
        fast_agent("Another quick call", user_id="user1", session_id="session1"),
        slow_agent("Another slow query", user_id="user2", session_id="session2")
    ]
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    # Run simulation to generate metrics
    asyncio.run(simulate_agent_calls())
    
    # Print collected metrics for verification
    print("Custom Metrics:", metrics_collector.get_all_metrics())
    
    # Instructions to run the server
    print("Run the server with: uvicorn examples.analytics_example:app.api --reload")