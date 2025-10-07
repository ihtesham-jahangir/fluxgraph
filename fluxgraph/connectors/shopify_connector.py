import aiohttp
from .base import BaseConnector

class ShopifyConnector(BaseConnector):
    """Shopify connector."""
    
    async def connect(self):
        self.shop_url = self.config['shop_url']
        self.access_token = self.config['access_token']
        self.connected = True
    
    async def disconnect(self):
        self.connected = False
    
    async def execute(self, operation: str, **kwargs):
        headers = {'X-Shopify-Access-Token': self.access_token}
        async with aiohttp.ClientSession() as session:
            if operation == 'get_products':
                url = f"{self.shop_url}/admin/api/2024-01/products.json"
                async with session.get(url, headers=headers) as resp:
                    data = await resp.json()
                    return data.get('products', [])
            elif operation == 'create_product':
                url = f"{self.shop_url}/admin/api/2024-01/products.json"
                async with session.post(url, headers=headers, json={'product': kwargs['data']}) as resp:
                    return await resp.json()
            elif operation == 'update_product':
                product_id = kwargs['product_id']
                url = f"{self.shop_url}/admin/api/2024-01/products/{product_id}.json"
                async with session.put(url, headers=headers, json={'product': kwargs['data']}) as resp:
                    return await resp.json()
            elif operation == 'delete_product':
                product_id = kwargs['product_id']
                url = f"{self.shop_url}/admin/api/2024-01/products/{product_id}.json"
                async with session.delete(url, headers=headers) as resp:
                    return resp.status == 200
            else:
                raise ValueError(f"Unsupported operation: {operation}")