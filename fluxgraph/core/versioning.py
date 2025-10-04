# fluxgraph/core/versioning.py
"""
Agent Versioning System for FluxGraph.
Enables A/B testing, rollbacks, and blue-green deployments.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


class AgentVersion:
    """Represents a versioned agent."""
    
    def __init__(
        self,
        agent_name: str,
        version: str,
        agent_code: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        self.agent_name = agent_name
        self.version = version
        self.agent_code = agent_code
        self.description = description
        self.config = config or {}
        self.tags = tags or []
        self.created_at = datetime.utcnow()
        self.checksum = self._calculate_checksum()
        self.deployment_count = 0
        self.active = False
    
    def _calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum of agent code."""
        return hashlib.sha256(self.agent_code.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "version": self.version,
            "description": self.description,
            "config": self.config,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "checksum": self.checksum,
            "deployment_count": self.deployment_count,
            "active": self.active
        }


class VersionedAgent:
    """Wrapper for agent with version management."""
    
    def __init__(self, base_agent: Any, version: AgentVersion):
        self.base_agent = base_agent
        self.version = version
    
    async def run(self, **kwargs):
        """Execute agent with version tracking."""
        logger.debug(f"Executing {self.version.agent_name} v{self.version.version}")
        return await self.base_agent.run(**kwargs)


class AgentVersionManager:
    """
    Manages agent versions with A/B testing and deployment strategies.
    """
    
    def __init__(self, storage_path: str = "./agent_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, List[AgentVersion]] = {}
        self.active_versions: Dict[str, str] = {}  # agent_name -> version
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"AgentVersionManager initialized at {storage_path}")
    
    def register_version(
        self,
        agent_name: str,
        version: str,
        agent_code: str,
        description: str,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        auto_activate: bool = False
    ) -> AgentVersion:
        """
        Register a new agent version.
        
        Args:
            agent_name: Name of the agent
            version: Version string (e.g., "1.0.0", "2.1.3")
            agent_code: Agent source code or serialized function
            description: Version description
            config: Optional configuration
            tags: Optional tags (e.g., ["beta", "experimental"])
            auto_activate: Automatically activate this version
        
        Returns:
            AgentVersion object
        """
        agent_version = AgentVersion(
            agent_name=agent_name,
            version=version,
            agent_code=agent_code,
            description=description,
            config=config,
            tags=tags
        )
        
        # Store version
        if agent_name not in self.versions:
            self.versions[agent_name] = []
        
        self.versions[agent_name].append(agent_version)
        
        # Save to disk
        self._save_version(agent_version)
        
        # Auto-activate if requested
        if auto_activate:
            self.activate_version(agent_name, version)
        
        logger.info(
            f"[Version] Registered {agent_name} v{version} "
            f"(Checksum: {agent_version.checksum[:8]}...)"
        )
        
        return agent_version
    
    def activate_version(self, agent_name: str, version: str):
        """
        Activate a specific version of an agent.
        
        Args:
            agent_name: Name of the agent
            version: Version to activate
        """
        versions = self.versions.get(agent_name, [])
        target_version = None
        
        for v in versions:
            if v.version == version:
                target_version = v
                v.active = True
            else:
                v.active = False
        
        if not target_version:
            raise ValueError(f"Version {version} not found for agent {agent_name}")
        
        self.active_versions[agent_name] = version
        target_version.deployment_count += 1
        
        logger.info(f"[Version] Activated {agent_name} v{version}")
    
    def get_active_version(self, agent_name: str) -> Optional[AgentVersion]:
        """Get the currently active version of an agent."""
        version_str = self.active_versions.get(agent_name)
        if not version_str:
            return None
        
        versions = self.versions.get(agent_name, [])
        for v in versions:
            if v.version == version_str:
                return v
        
        return None
    
    def rollback(self, agent_name: str) -> Optional[str]:
        """
        Rollback to the previous version.
        
        Args:
            agent_name: Name of the agent
        
        Returns:
            Version string of rolled back version, or None
        """
        versions = self.versions.get(agent_name, [])
        if len(versions) < 2:
            logger.warning(f"Cannot rollback {agent_name}: not enough versions")
            return None
        
        # Sort by creation date
        sorted_versions = sorted(versions, key=lambda v: v.created_at, reverse=True)
        
        # Get previous version (second in list)
        previous_version = sorted_versions[1]
        
        self.activate_version(agent_name, previous_version.version)
        
        logger.info(
            f"[Version] Rolled back {agent_name} to v{previous_version.version}"
        )
        
        return previous_version.version
    
    def start_ab_test(
        self,
        agent_name: str,
        version_a: str,
        version_b: str,
        split_ratio: float = 0.5,
        metric: str = "success_rate"
    ) -> str:
        """
        Start A/B test between two agent versions.
        
        Args:
            agent_name: Name of the agent
            version_a: First version (control)
            version_b: Second version (variant)
            split_ratio: Traffic split (0.5 = 50/50)
            metric: Metric to compare
        
        Returns:
            A/B test ID
        """
        test_id = f"{agent_name}_ab_{datetime.utcnow().timestamp()}"
        
        self.ab_tests[test_id] = {
            "agent_name": agent_name,
            "version_a": version_a,
            "version_b": version_b,
            "split_ratio": split_ratio,
            "metric": metric,
            "started_at": datetime.utcnow().isoformat(),
            "results_a": [],
            "results_b": []
        }
        
        logger.info(
            f"[ABTest:{test_id}] Started: {agent_name} v{version_a} vs v{version_b} "
            f"(Split: {split_ratio:.0%})"
        )
        
        return test_id
    
    def select_ab_version(self, test_id: str, user_id: str) -> str:
        """
        Select which version to use for a user in A/B test.
        
        Args:
            test_id: A/B test identifier
            user_id: User identifier (for consistent routing)
        
        Returns:
            Version string to use
        """
        test = self.ab_tests.get(test_id)
        if not test:
            raise ValueError(f"A/B test {test_id} not found")
        
        # Use hash for consistent user routing
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        use_variant_b = (user_hash % 100) < (test["split_ratio"] * 100)
        
        return test["version_b"] if use_variant_b else test["version_a"]
    
    def record_ab_result(
        self,
        test_id: str,
        version: str,
        result: Dict[str, Any]
    ):
        """Record A/B test result."""
        test = self.ab_tests.get(test_id)
        if not test:
            return
        
        if version == test["version_a"]:
            test["results_a"].append(result)
        else:
            test["results_b"].append(result)
    
    def get_ab_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results."""
        test = self.ab_tests.get(test_id)
        if not test:
            return {"error": f"Test {test_id} not found"}
        
        results_a = test["results_a"]
        results_b = test["results_b"]
        
        return {
            "test_id": test_id,
            "version_a": {
                "version": test["version_a"],
                "samples": len(results_a),
                "results": results_a
            },
            "version_b": {
                "version": test["version_b"],
                "samples": len(results_b),
                "results": results_b
            }
        }
    
    def _save_version(self, version: AgentVersion):
        """Save version to disk."""
        version_path = self.storage_path / version.agent_name
        version_path.mkdir(exist_ok=True)
        
        file_path = version_path / f"{version.version}.json"
        
        with open(file_path, 'w') as f:
            json.dump({
                **version.to_dict(),
                "agent_code": version.agent_code
            }, f, indent=2)
    
    def list_versions(self, agent_name: str) -> List[Dict[str, Any]]:
        """List all versions of an agent."""
        versions = self.versions.get(agent_name, [])
        return [v.to_dict() for v in versions]
