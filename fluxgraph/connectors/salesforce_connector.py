import aiohttp
from .base import BaseConnector

class SalesforceConnector(BaseConnector):
    """Salesforce CRM connector."""
    
    async def connect(self):
        self.instance_url = self.config['instance_url']
        self.access_token = await self._authenticate()
        self.connected = True
    
    async def _authenticate(self):
        auth_url = f"{self.config['instance_url']}/services/oauth2/token"
        data = {
            'grant_type': 'password',
            'client_id': self.config['client_id'],
            'client_secret': self.config['client_secret'],
            'username': self.config['username'],
            'password': self.config['password']
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(auth_url, data=data) as resp:
                result = await resp.json()
                return result['access_token']
    
    async def disconnect(self):
        self.connected = False
    
    async def execute(self, operation: str, **kwargs):
        headers = {'Authorization': f'Bearer {self.access_token}', 'Content-Type': 'application/json'}
        async with aiohttp.ClientSession() as session:
            if operation == 'create_lead':
                url = f"{self.instance_url}/services/data/v57.0/sobjects/Lead/"
                async with session.post(url, headers=headers, json=kwargs['data']) as resp:
                    return await resp.json()
            elif operation == 'get_lead':
                lead_id = kwargs['lead_id']
                url = f"{self.instance_url}/services/data/v57.0/sobjects/Lead/{lead_id}"
                async with session.get(url, headers=headers) as resp:
                    return await resp.json()
            elif operation == 'update_lead':
                lead_id = kwargs['lead_id']
                url = f"{self.instance_url}/services/data/v57.0/sobjects/Lead/{lead_id}"
                async with session.patch(url, headers=headers, json=kwargs['data']) as resp:
                    return resp.status == 204
            elif operation == 'delete_lead':
                lead_id = kwargs['lead_id']
                url = f"{self.instance_url}/services/data/v57.0/sobjects/Lead/{lead_id}"
                async with session.delete(url, headers=headers) as resp:
                    return resp.status == 204
            else:
                raise ValueError(f"Unsupported operation: {operation}")