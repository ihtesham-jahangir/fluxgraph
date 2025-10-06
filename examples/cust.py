import os
import aiohttp
import logging
import re
import asyncpg
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fluxgraph import FluxApp

TINYLAMA_API = os.getenv("TINYLAMA_API_URL", "https://tinyllm.alphanetwork.com.pk")
TINYLAMA_USER = os.getenv("TINYLAMA_USER", "ihteshamjahangir21@gmail.com")
TINYLAMA_PASS = os.getenv("TINYLAMA_PASS", "123456")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_Qjrwmf6u8OWD@ep-summer-art-ad4wfcr3-pooler.c-2.us-east-1.aws.neon.tech/fluxgraph?sslmode=require&channel_binding=require")

class TinyLlamaLLM:
    """TinyLlama API client with JWT authentication and aggressive response cleaning."""
    def __init__(self):
        self.api = TINYLAMA_API
        self.user = TINYLAMA_USER
        self.password = TINYLAMA_PASS
        self.token = None

    async def login(self):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self.api}/login",
                json={"identifier": self.user, "password": self.password}
            )
            resp.raise_for_status()
            data = await resp.json()
            self.token = data["access_token"]

    async def generate(self, prompt: str, max_tokens=200, temperature=0.7) -> str:
        if not self.token:
            await self.login()
        headers = {"Authorization": f"Bearer {self.token}"}
        body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.api}/chat", headers=headers, json=body)
            if resp.status == 401:
                await self.login()
                headers["Authorization"] = f"Bearer {self.token}"
                resp = await session.post(f"{self.api}/chat", headers=headers, json=body)
            resp.raise_for_status()
            data = await resp.json()
            raw_text = data.get("text", "")
            response = self.clean_response(raw_text)
            return response
    
    def clean_response(self, text: str) -> str:
        """Aggressively clean LLM response to get only the actual answer."""
        markers = ["Assistant:", "Response:", "Answer:", "Reply:", "Agent:", "Bot:", "AI:", "Output:"]
        for marker in markers:
            if marker in text:
                text = text.split(marker)[-1]
        
        artifacts = [
            "Previous messages:", "User:", "Customer:", "You are",
            "Task:", "Context:", "Conversation history:", 
            "Recent conversation:", "Customer info:", "Customer profile:",
            "hide previous messages", "experiencing"
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if any(artifact.lower() in line.lower() for artifact in artifacts):
                continue
            if len(line) < 10:
                continue
            if line.endswith(':') and len(line) < 30:
                continue
            cleaned_lines.append(line)
        
        response = ' '.join(cleaned_lines).strip()
        response = re.sub(r'^[:\-\.\,\s]+', '', response)
        response = re.sub(r'[:\-\s]+$', '', response)
        
        if len(response) < 15:
            return ""
        
        if len(response) > 3500:
            sentences = response.split('.')
            response = '. '.join(sentences[:2]) + '.'
        
        return response.strip()

llm = TinyLlamaLLM()

# Database connection pool
db_pool = None

async def get_db_pool():
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)
    return db_pool

async def get_products():
    """Fetch all products from database."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, name, description, price FROM products ORDER BY id")
        return [dict(row) for row in rows]

async def search_products(query: str):
    """Search products by name or description."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, description, price FROM products WHERE name ILIKE $1 OR description ILIKE $1",
            f"%{query}%"
        )
        return [dict(row) for row in rows]

app = FluxApp(
    title="Alpha Virtual Assistant",
    version="3.0.0",
    enable_workflows=False,
    enable_advanced_memory=False,
    enable_agent_cache=False,
    enable_streaming=True,
    enable_security=False,
)

app.api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversations = {}

@app.agent(name="enterprise_virtual_agent")
async def virtual_agent(
    message: str,
    session_id: str = None,
    **kwargs
):
    """Human-like AI agent with product database integration."""
    
    if session_id not in conversations:
        conversations[session_id] = []
    
    history = conversations[session_id]
    lower_msg = message.lower()
    
    # Detect intent
    intent = "general"
    products_info = ""
    
    # Check if asking about products
    if any(w in lower_msg for w in ["product", "products", "laptop", "phone", "headphone", "price", "cost", "buy", "purchase", "how much", "available"]):
        intent = "product_inquiry"
        
        # Get products from database
        try:
            if any(w in lower_msg for w in ["laptop", "gaming"]):
                products = await search_products("laptop")
            elif any(w in lower_msg for w in ["phone", "smartphone"]):
                products = await search_products("phone")
            elif any(w in lower_msg for w in ["headphone", "audio"]):
                products = await search_products("headphone")
            else:
                products = await get_products()
            
            if products:
                products_info = "\n\nAvailable Products:\n"
                for p in products:
                    products_info += f"- {p['name']}: {p['description']} - ${p['price']}\n"
        except Exception as e:
            logging.error(f"Database error: {e}")
            products_info = ""
    
    # Other intent detection
    elif any(w in lower_msg for w in ["human", "agent", "real person", "manager"]):
        intent = "escalation"
    elif any(w in lower_msg for w in ["bye", "goodbye", "thanks", "thank you"]):
        intent = "farewell"
    elif any(w in lower_msg for w in ["hi", "hello", "hey"]):
        intent = "greeting"
    elif any(w in lower_msg for w in ["help", "support", "problem", "issue", "order"]):
        intent = "support"
    
    # Handle escalation
    if intent == "escalation":
        response = "I'll connect you with a human agent right away. Please hold on."
        conversations[session_id].append(message)
        conversations[session_id].append(response)
        return {
            "intent": "escalation",
            "data": {"query": message, "priority": "high", "status": "escalated"},
            "action": "handover_to_agent",
            "response": response,
        }
    
    # Build prompt with product info if available
    if intent == "product_inquiry" and products_info:
        prompt = f"You are Alpha Networks sales assistant. {products_info}\n\nCustomer: {message}\n\nProvide helpful product information and pricing.\n\nYou:"
    elif intent == "greeting":
        prompt = f"Greet the customer warmly and ask how you can help.\nCustomer: {message}\nYou:"
    elif intent == "support":
        prompt = f"Customer needs support. Be empathetic and helpful.\nCustomer: {message}\nYou:"
    elif intent == "farewell":
        prompt = f"Customer is saying goodbye. Thank them warmly.\nCustomer: {message}\nYou:"
    else:
        prompt = f"Respond naturally and helpfully.\nCustomer: {message}\nYou:"
    
    # Get LLM response
    try:
        response = await llm.generate(prompt, max_tokens=150, temperature=0.7)
        
        if not response or len(response) < 15:
            fallbacks = {
                "greeting": "Hello! I'm here to help you with Alpha Networks services. What can I assist you with today?",
                "product_inquiry": f"Here are our available products:{products_info}\n\nWhich one interests you?",
                "support": "I'm here to help resolve your issue. Could you tell me more about what's happening?",
                "farewell": "Thank you for reaching out! Feel free to contact us anytime.",
                "general": "I'm here to help! How can I assist you today?"
            }
            response = fallbacks.get(intent, fallbacks["general"])
        
        bad_phrases = ["Previous messages", "User:", "Customer:", "You are", "Task:", "hide previous"]
        for phrase in bad_phrases:
            if phrase in response:
                fallbacks = {
                    "greeting": "Hi there! How can I help you today?",
                    "product_inquiry": f"We have great products available:{products_info}",
                    "support": "I'm here to help. What issue are you experiencing?",
                    "farewell": "Thanks for chatting! Have a great day!",
                    "general": "How can I assist you?"
                }
                response = fallbacks.get(intent, "How can I help you?")
                break
                
    except Exception as e:
        logging.error(f"LLM error: {e}")
        response = "I'm having a small technical issue. Could you please rephrase that?"
    
    priority = "high" if any(w in lower_msg for w in ["urgent", "immediately", "asap"]) else "normal"
    
    conversations[session_id].append(message)
    conversations[session_id].append(response)
    if len(conversations[session_id]) > 10:
        conversations[session_id] = conversations[session_id][-10:]
    
    return {
        "intent": intent,
        "data": {
            "query": message,
            "priority": priority,
            "status": "open",
        },
        "action": "respond_to_user",
        "response": response,
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
