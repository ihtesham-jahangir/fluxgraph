Agent Development
=================

Creating Agents
---------------

Agents are the core building blocks of FluxGraph:

.. code-block:: python

   @app.agent(name="my_agent")
   async def my_agent(
       message: str,
       session_id: str = None,
       context: dict = None,
       memory=None,
       advanced_memory=None,
       cache=None,
       tools=None,
       rag=None,
       connectors=None
   ):
       return {"response": "Hello!"}


Agent Parameters
----------------

**Required:**
- ``message`` (str): User input message

**Optional:**
- ``session_id`` (str): Unique session identifier
- ``context`` (dict): Additional context data
- ``memory``: Basic memory interface
- ``advanced_memory``: Advanced memory with recall
- ``cache``: Agent cache for performance
- ``tools``: Registered tool functions
- ``rag``: RAG retrieval system
- ``connectors``: External API/DB connectors


Agent Response Format
---------------------

.. code-block:: python

   return {
       "response": "Agent reply text",
       "intent": "user_intent",
       "data": {...},
       "action": "respond_to_user",
       "status": "success"
   }


Multi-Agent Orchestration
--------------------------

Call other agents from within an agent:

.. code-block:: python

   @app.agent(name="orchestrator")
   async def orchestrator(message: str, call_agent=None):
       # Call first agent
       result1 = await call_agent("agent1", {"message": message})
       
       # Call second agent with result
       result2 = await call_agent("agent2", {
           "message": result1["response"]
       })
       
       return result2


Intent Detection
----------------

.. code-block:: python

   def detect_intent(message: str) -> str:
       lower_msg = message.lower()
       
       if any(w in lower_msg for w in ["buy", "price", "cost"]):
           return "sales"
       elif any(w in lower_msg for w in ["help", "support"]):
           return "support"
       else:
           return "general"


Error Handling
--------------

.. code-block:: python

   import logging

   @app.agent(name="safe_agent")
   async def safe_agent(message: str):
       try:
           response = await process_message(message)
           return {"response": response, "status": "success"}
       except Exception as e:
           logging.error(f"Agent error: {e}")
           return {
               "response": "I encountered an error.",
               "status": "error",
               "error": str(e)
           }
