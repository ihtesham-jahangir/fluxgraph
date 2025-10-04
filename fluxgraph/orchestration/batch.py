# fluxgraph/orchestration/batch.py
"""
Batch Processing System for FluxGraph.
Handles large-scale asynchronous agent task processing.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class BatchStatus(Enum):
    """Batch job status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL_FAILURE = "partial_failure"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchTask:
    """Represents a single task in a batch."""
    
    def __init__(self, task_id: str, agent_name: str, payload: Dict[str, Any]):
        self.task_id = task_id
        self.agent_name = agent_name
        self.payload = payload
        self.status = "pending"
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None


class BatchJob:
    """Represents a batch processing job."""
    
    def __init__(
        self,
        job_id: str,
        tasks: List[BatchTask],
        priority: int = 0,
        max_concurrent: int = 10
    ):
        self.job_id = job_id
        self.tasks = tasks
        self.priority = priority
        self.max_concurrent = max_concurrent
        self.status = BatchStatus.QUEUED
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.progress = 0
    
    def get_summary(self) -> Dict[str, Any]:
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total_tasks": len(self.tasks),
            "completed": completed,
            "failed": failed,
            "progress": (completed / len(self.tasks) * 100) if self.tasks else 0,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


class BatchProcessor:
    """
    Processes batch jobs asynchronously with concurrency control.
    Supports priority queues and progress tracking.
    """
    
    def __init__(self, orchestrator, max_concurrent_jobs: int = 5):
        self.orchestrator = orchestrator
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: List[BatchJob] = []
        self._processing = False
        logger.info(f"BatchProcessor initialized (max concurrent: {max_concurrent_jobs})")
    
    async def submit_batch(
        self,
        agent_name: str,
        payloads: List[Dict[str, Any]],
        priority: int = 0,
        max_concurrent: int = 10
    ) -> str:
        """
        Submit a batch processing job.
        
        Args:
            agent_name: Name of agent to execute
            payloads: List of payloads for each task
            priority: Job priority (lower = higher priority)
            max_concurrent: Max concurrent tasks within this job
        
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        # Create tasks
        tasks = [
            BatchTask(
                task_id=f"{job_id}_{i}",
                agent_name=agent_name,
                payload=payload
            )
            for i, payload in enumerate(payloads)
        ]
        
        # Create job
        job = BatchJob(
            job_id=job_id,
            tasks=tasks,
            priority=priority,
            max_concurrent=max_concurrent
        )
        
        # Queue job
        await self.job_queue.put((priority, job))
        
        logger.info(
            f"[Batch:{job_id[:8]}] Submitted {len(tasks)} tasks "
            f"for agent '{agent_name}' (Priority: {priority})"
        )
        
        # Start processor if not running
        if not self._processing:
            asyncio.create_task(self._process_queue())
        
        return job_id
    
    async def _process_queue(self):
        """Process jobs from the queue."""
        self._processing = True
        
        try:
            while not self.job_queue.empty():
                # Check concurrent job limit
                if len(self.active_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(1)
                    continue
                
                # Get next job
                priority, job = await self.job_queue.get()
                
                # Process job
                asyncio.create_task(self._process_job(job))
        
        finally:
            self._processing = False
    
    async def _process_job(self, job: BatchJob):
        """Process a single batch job."""
        self.active_jobs[job.job_id] = job
        job.status = BatchStatus.PROCESSING
        job.started_at = datetime.utcnow()
        
        logger.info(
            f"[Batch:{job.job_id[:8]}] Started processing {len(job.tasks)} tasks "
            f"(Max concurrent: {job.max_concurrent})"
        )
        
        # Process tasks with concurrency limit
        semaphore = asyncio.Semaphore(job.max_concurrent)
        
        async def process_task(task: BatchTask):
            async with semaphore:
                await self._execute_task(task)
        
        # Execute all tasks
        await asyncio.gather(*[process_task(task) for task in job.tasks])
        
        # Update job status
        failed_tasks = sum(1 for t in job.tasks if t.status == "failed")
        
        if failed_tasks == 0:
            job.status = BatchStatus.COMPLETED
        elif failed_tasks == len(job.tasks):
            job.status = BatchStatus.FAILED
        else:
            job.status = BatchStatus.PARTIAL_FAILURE
        
        job.completed_at = datetime.utcnow()
        
        # Move to completed
        self.active_jobs.pop(job.job_id)
        self.completed_jobs.append(job)
        
        duration = (job.completed_at - job.started_at).total_seconds()
        
        logger.info(
            f"[Batch:{job.job_id[:8]}] Completed with status {job.status.value} "
            f"in {duration:.2f}s (Failed: {failed_tasks}/{len(job.tasks)})"
        )
    
    async def _execute_task(self, task: BatchTask):
        """Execute a single task."""
        task.started_at = datetime.utcnow()
        task.status = "processing"
        
        try:
            result = await self.orchestrator.run(task.agent_name, task.payload)
            task.result = result
            task.status = "completed"
        
        except Exception as e:
            task.error = str(e)
            task.status = "failed"
            logger.error(f"[Batch:Task:{task.task_id}] Failed: {e}")
        
        finally:
            task.completed_at = datetime.utcnow()
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a batch job."""
        # Check active jobs
        job = self.active_jobs.get(job_id)
        if job:
            return job.get_summary()
        
        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return job.get_summary()
        
        return None
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job."""
        job = self.active_jobs.get(job_id)
        if job:
            job.status = BatchStatus.CANCELLED
            logger.info(f"[Batch:{job_id[:8]}] Cancelled")
            return True
        return False
