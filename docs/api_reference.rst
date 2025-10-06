API Reference
=============

FluxApp
-------

.. autoclass:: fluxgraph.FluxApp
   :members:
   :undoc-members:


Agent Decorator
---------------

.. code-block:: python

   @app.agent(name: str)
   async def agent_function(**kwargs):
       pass


Endpoints
---------

**Execute Agent**

.. code-block:: http

   POST /ask/{agent_name}
   Content-Type: application/json

   {
     "message": "User message",
     "session_id": "optional_session_id"
   }

**Response:**

.. code-block:: json

   {
     "response": "Agent response",
     "intent": "detected_intent",
     "action": "respond_to_user"
   }


Advanced Memory
---------------

.. autoclass:: fluxgraph.core.advanced_memory.AdvancedMemory
   :members: store, recall, recall_recent
