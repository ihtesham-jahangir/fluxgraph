Complete Examples
=================

Lead Generation Agent
---------------------

.. code-block:: python

   from fluxgraph import FluxApp
   import asyncpg

   app = FluxApp(title="Lead Gen Agent", version="1.0.0")

   @app.agent(name="lead_qualifier")
   async def lead_qualifier(
       message: str,
       session_id: str = None,
       advanced_memory=None
   ):
       # Intent detection
       lower_msg = message.lower()
       if any(w in lower_msg for w in ["pricing", "demo", "interested"]):
           intent = "hot_lead"
       else:
           intent = "qualifying"
       
       # Extract contact info
       import re
       email = re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', message)
       
       response = {
           "intent": intent,
           "response": "I'd love to help! What services are you interested in?",
           "data": {
               "email": email.group() if email else None,
               "priority": "high" if intent == "hot_lead" else "normal"
           }
       }
       
       # Store in memory
       if advanced_memory:
           await advanced_memory.store(
               f"Lead: {message}",
               metadata={"session_id": session_id, "intent": intent}
           )
       
       return response


Product Search Agent with PostgreSQL
-------------------------------------

.. code-block:: python

   import asyncpg

   DATABASE_URL = "postgresql://user:pass@localhost/db"
   db_pool = None

   async def get_db():
       global db_pool
       if not db_pool:
           db_pool = await asyncpg.create_pool(DATABASE_URL)
       return db_pool

   @app.agent(name="product_agent")
   async def product_agent(message: str):
       pool = await get_db()
       async with pool.acquire() as conn:
           products = await conn.fetch(
               "SELECT name, price FROM products WHERE name ILIKE $1",
               f"%{message}%"
           )
       
       if products:
           product_list = "\n".join([
               f"- {p['name']}: ${p['price']}"
               for p in products
           ])
           response = f"Here are our products:\n{product_list}"
       else:
           response = "Sorry, no products found."
       
       return {"response": response}


Customer Support with Escalation
---------------------------------

.. code-block:: python

   @app.agent(name="support_agent")
   async def support_agent(message: str, advanced_memory=None):
       lower_msg = message.lower()
       
       # Check for escalation keywords
       if any(w in lower_msg for w in ["human", "agent", "manager"]):
           return {
               "response": "Connecting you to a human agent...",
               "action": "handover_to_agent",
               "priority": "high"
           }
       
       # Handle support query
       if "order" in lower_msg:
           response = "Please provide your order ID."
       else:
           response = "I'm here to help! What issue are you experiencing?"
       
       return {"response": response, "action": "respond_to_user"}
