# examples/full_v31_example.py
"""
FluxGraph v3.1 Complete Example with Visual Workflow Builder Integration
"""
import asyncio
from fluxgraph import FluxApp

# Initialize with ALL v3.1 features
app = FluxApp(
    title="Alpha Virtual Assistant v3.1",
    version="3.1.0",
    # v3.1 features
    enable_connectors=True,
    enable_enhanced_memory=True,
    enable_visual_workflows=True,
    # Database URL for enhanced memory
    database_url="postgresql://mag_owner:npg_Z3dEeUNa7Amq@ep-floral-voice-a87kfg2z-pooler.eastus2.azure.neon.tech/mag?sslmode=require&channel_binding=require",
    # v3.0 features
    enable_agent_cache=True,
    enable_advanced_memory=True,
    enable_workflows=True,
    # Enterprise features
    enable_analytics=True,
    enable_advanced_features=True
)

# Add PostgreSQL connector
app.add_connector('products_db', 'postgres', {
    'url': 'postgresql://mag_owner:npg_Z3dEeUNa7Amq@ep-floral-voice-a87kfg2z-pooler.eastus2.azure.neon.tech/mag?sslmode=require&channel_binding=require'
})

# ===== AGENTS =====

@app.agent(name="greeting")
async def greeting_agent(message: str = "", **kwargs):
    """Simple greeting agent for testing."""
    return {
        "response": f"Hello! You said: {message}",
        "status": "success",
        "timestamp": "2025-10-06T12:00:00Z"
    }

@app.agent(name="product_search")
async def product_search_agent(
    message: str = "",
    session_id: str = None,
    connectors=None,
    enhanced_memory=None,
    **kwargs
):
    """Search products in database."""
    
    # Get database connector
    if connectors:
        try:
            db = await connectors.get_connector('products_db')
            
            # Search products
            products = await db.execute('query',
                sql="SELECT * FROM products WHERE name ILIKE $1 OR description ILIKE $1 LIMIT 10",
                params=[f"%{message}%"]
            )
            
            if products:
                response = "Here are the products I found:\n"
                for p in products:
                    response += f"- {p['name']}: ${p['price']}\n"
                
                # Store in enhanced memory
                if enhanced_memory and session_id:
                    await enhanced_memory.store_conversation(
                        session_id=session_id,
                        agent_name="product_search",
                        user_message=message,
                        agent_response=response,
                        intent="product_search",
                        metadata={"products_found": len(products)}
                    )
                
                return {
                    "response": response,
                    "products": [dict(p) for p in products],
                    "count": len(products),
                    "status": "success"
                }
            else:
                return {
                    "response": "No products found matching your query.",
                    "products": [],
                    "count": 0,
                    "status": "success"
                }
        except Exception as e:
            return {
                "response": f"Error searching products: {str(e)}",
                "status": "error",
                "error": str(e)
            }
    
    return {
        "response": "Database connector not available",
        "status": "error"
    }

@app.agent(name="customer_support")
async def customer_support_agent(
    message: str = "",
    session_id: str = None,
    **kwargs
):
    """Handle customer support queries."""
    response = f"Thank you for contacting support. Regarding your query: '{message}'\n\n"
    response += "Our team will assist you shortly. Your ticket number is: #CS-2025-001"
    
    return {
        "response": response,
        "ticket_id": "CS-2025-001",
        "status": "pending",
        "priority": "medium"
    }

@app.agent(name="order_status")
async def order_status_agent(
    order_id: str = "",
    message: str = "",
    **kwargs
):
    """Check order status."""
    return {
        "response": f"Order {order_id or 'N/A'} is currently being processed.",
        "order_id": order_id or "N/A",
        "status": "processing",
        "estimated_delivery": "2025-10-10",
        "tracking_number": "TRK123456789"
    }

# ===== VISUAL WORKFLOWS =====

# Create product search workflow
product_workflow = app.create_visual_workflow("product_search_workflow")
search_node = product_workflow.add_node('agent', {'agent_name': 'product_search'})
product_workflow.start_node = search_node

# Create customer support workflow
support_workflow = app.create_visual_workflow("support_workflow")
support_node = support_workflow.add_node('agent', {'agent_name': 'customer_support'})
support_workflow.start_node = support_node

# ===== STARTUP INFO =====

print("\n" + "="*80)
print("✅ FluxGraph v3.1 - Alpha Virtual Assistant Ready")
print("="*80)
print(f"\n📊 Configuration:")
print(f"   - Title: {app.title}")
print(f"   - Version: {app.version}")
print(f"   - Enhanced Memory: {'ON' if app.enhanced_memory else 'OFF'}")
print(f"   - Connectors: {'ON' if app.connectors else 'OFF'}")
print(f"   - Visual Workflows: {'ON' if app.visual_workflows else 'OFF'}")

print(f"\n🔌 Active Connectors: {list(app.connectors.keys())}")
# ✅ FIXED: Use list_agents() method or _agents attribute
print(f"🤖 Registered Agents: {app.registry.list_agents()}")
print(f"🌐 Visual Workflows: {list(app.visual_workflows.keys())}")

print(f"\n💡 Available API Endpoints:")
print(f"   GET  http://localhost:8000/")
print(f"   POST http://localhost:8000/ask/greeting")
print(f"   POST http://localhost:8000/ask/product_search")
print(f"   POST http://localhost:8000/ask/customer_support")
print(f"   POST http://localhost:8000/ask/order_status")
print(f"   GET  http://localhost:8000/visual-workflows")
print(f"   POST http://localhost:8000/visual-workflows/product_search_workflow/execute")
print(f"   GET  http://localhost:8000/connectors")

print(f"\n🎨 Frontend:")
print(f"   http://localhost:3000 - Visual Workflow Builder")

print(f"\n📋 Example curl commands:")
print(f"   curl -X POST http://localhost:8000/ask/greeting \\")
print(f"     -H 'Content-Type: application/json' \\")
print(f"     -d '{{\"message\": \"Hello World\"}}'")

print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, reload=False)
