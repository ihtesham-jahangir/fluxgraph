# fluxgraph/security/rbac.py
"""
Role-Based Access Control (RBAC) for FluxGraph.
Provides authentication and authorization for agent access.
"""

import jwt
import secrets
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
from passlib.hash import bcrypt

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles with different permission levels."""
    ADMIN = "admin"
    DEVELOPER = "developer"
    USER = "user"
    READONLY = "readonly"


class Permission(Enum):
    """Granular permissions for actions."""
    EXECUTE_AGENT = "execute_agent"
    CREATE_AGENT = "create_agent"
    DELETE_AGENT = "delete_agent"
    VIEW_AGENTS = "view_agents"
    VIEW_COSTS = "view_costs"
    MANAGE_USERS = "manage_users"
    VIEW_LOGS = "view_logs"
    STREAM_RESPONSES = "stream_responses"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.EXECUTE_AGENT,
        Permission.CREATE_AGENT,
        Permission.DELETE_AGENT,
        Permission.VIEW_AGENTS,
        Permission.VIEW_COSTS,
        Permission.MANAGE_USERS,
        Permission.VIEW_LOGS,
        Permission.STREAM_RESPONSES,
    },
    Role.DEVELOPER: {
        Permission.EXECUTE_AGENT,
        Permission.CREATE_AGENT,
        Permission.VIEW_AGENTS,
        Permission.VIEW_COSTS,
        Permission.VIEW_LOGS,
        Permission.STREAM_RESPONSES,
    },
    Role.USER: {
        Permission.EXECUTE_AGENT,
        Permission.VIEW_AGENTS,
        Permission.STREAM_RESPONSES,
    },
    Role.READONLY: {
        Permission.VIEW_AGENTS,
    }
}


class User:
    """Represents a system user with roles and permissions."""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        email: str,
        roles: List[Role],
        api_key: Optional[str] = None
    ):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.roles = roles
        self.api_key = api_key or self._generate_api_key()
        self.created_at = datetime.utcnow()
        self.last_login: Optional[datetime] = None
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        return f"fluxgraph_{secrets.token_urlsafe(32)}"
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True
        return False
    
    def has_any_role(self, *roles: Role) -> bool:
        """Check if user has any of the specified roles."""
        return any(role in self.roles for role in roles)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": [r.value for r in self.roles],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


class RBACManager:
    """Manages users, roles, and permissions."""
    
    def __init__(self, jwt_secret: Optional[str] = None):
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.api_key_to_user: Dict[str, str] = {}
        logger.info("RBACManager initialized")
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: List[Role]
    ) -> User:
        """Create a new user."""
        user_id = secrets.token_urlsafe(16)
        password_hash = bcrypt.hash(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles
        )
        
        self.users[user_id] = user
        self.api_key_to_user[user.api_key] = user_id
        
        logger.info(f"Created user: {username} ({user_id}) with roles: {[r.value for r in roles]}")
        return user
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate a user by API key."""
        user_id = self.api_key_to_user.get(api_key)
        if not user_id:
            logger.warning(f"Authentication failed: Invalid API key")
            return None
        
        user = self.users.get(user_id)
        if user:
            user.last_login = datetime.utcnow()
            logger.info(f"User authenticated: {user.username}")
        return user
    
    def generate_jwt_token(self, user: User, expires_in_hours: int = 24) -> str:
        """Generate a JWT token for a user."""
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "roles": [r.value for r in user.roles],
            "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm="HS256")
        logger.info(f"Generated JWT token for user: {user.username}")
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def check_permission(self, user: User, permission: Permission) -> bool:
        """Check if a user has a specific permission."""
        has_perm = user.has_permission(permission)
        if not has_perm:
            logger.warning(
                f"Permission denied: {user.username} does not have {permission.value}"
            )
        return has_perm
    
    def revoke_api_key(self, user_id: str):
        """Revoke a user's API key and generate a new one."""
        user = self.users.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Remove old API key
        self.api_key_to_user.pop(user.api_key, None)
        
        # Generate new API key
        user.api_key = user._generate_api_key()
        self.api_key_to_user[user.api_key] = user_id
        
        logger.info(f"API key revoked and regenerated for user: {user.username}")
