Quick Start Guide
=================

Installation
------------

.. code-block:: bash

   pip install fluxgraph

For development:

.. code-block:: bash

   git clone https://github.com/yourusername/fluxgraph.git
   cd fluxgraph
   pip install -e .


Basic Agent Example
-------------------

Create a simple conversational agent:

.. code-block:: python

   from fluxgraph import FluxApp

   app = FluxApp(
       title="Customer Support Agent",
       version="1.0.0",
       enable_advanced_memory=True
   )

   @app.agent(name="support_agent")
   async def support_agent(
       message: str,
       session_id: str = None,
       advanced_memory=None
   ):
       # Store conversation in memory
       if advanced_memory:
           await advanced_memory.store(
               message,
               memory_type="episodic",
               metadata={"session_id": session_id}
           )
       
       return {
           "response": "I'm here to help! How can I assist you?",
           "status": "success"
       }

   if __name__ == "__main__":
       app.run(host="0.0.0.0", port=8000)


Running the Agent
-----------------

Start your agent server:

.. code-block:: bash

   python app.py

Access the API:

.. code-block:: bash

   curl -X POST http://localhost:8000/ask/support_agent \
     -H "Content-Type: application/json" \
     -d '{"message": "Hello!", "session_id": "user123"}'


Using LLM Providers
-------------------

Integrate TinyLlama:

.. code-block:: python

   import aiohttp

   class TinyLlamaLLM:
       async def generate(self, prompt: str):
           async with aiohttp.ClientSession() as session:
               resp = await session.post(
                   "https://tinyllm.alphanetwork.com.pk/chat",
                   headers={"Authorization": f"Bearer {token}"},
                   json={"prompt": prompt, "max_tokens": 200}
               )
               data = await resp.json()
               return data["text"]

   llm = TinyLlamaLLM()

   @app.agent(name="ai_agent")
   async def ai_agent(message: str):
       response = await llm.generate(f"User: {message}\nAssistant:")
       return {"response": response}


Next Steps
----------

- Learn about :doc:`agents` and orchestration
- Explore :doc:`memory` systems
- Build :doc:`workflows`
- Deploy to :doc:`deployment`
