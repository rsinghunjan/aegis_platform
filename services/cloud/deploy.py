"""Model deployment abstraction across cloud compute targets."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class DeploymentTarget(str, Enum):
    EC2 = "ec2"
    CLOUD_RUN = "cloud_run"
    AZURE_CONTAINER_INSTANCES = "azure_container_instances"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class DeploymentSpec:
    model_name: str
    model_version: str
    target: DeploymentTarget
    replicas: int = 1
    cpu: str = "1"
    memory: str = "2Gi"
    gpu: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)


@dataclass
class Deployment:
    deployment_id: str
    spec: DeploymentSpec
    status: DeploymentStatus
    endpoint: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class ModelDeployment:
    """Lifecycle manager for model deployments across compute targets.

    Actual provisioning calls are deferred to target-specific backends
    (not implemented here to avoid hard SDK dependencies); this class
    manages deployment state transitions and provides the integration
    point where a real backend would be plugged in via ``deploy_fn``.
    """

    def __init__(self):
        self._deployments: Dict[str, Deployment] = {}

    def deploy(self, spec: DeploymentSpec, deploy_fn: Optional[Any] = None) -> Deployment:
        deployment_id = str(uuid.uuid4())
        deployment = Deployment(
            deployment_id=deployment_id, spec=spec, status=DeploymentStatus.DEPLOYING
        )
        self._deployments[deployment_id] = deployment
        try:
            endpoint = deploy_fn(spec) if deploy_fn else f"local://{spec.model_name}:{spec.model_version}"
            deployment.endpoint = endpoint
            deployment.status = DeploymentStatus.RUNNING
        except Exception:
            deployment.status = DeploymentStatus.FAILED
            raise
        return deployment

    def get(self, deployment_id: str) -> Deployment:
        if deployment_id not in self._deployments:
            raise KeyError(deployment_id)
        return self._deployments[deployment_id]

    def terminate(self, deployment_id: str) -> Deployment:
        deployment = self.get(deployment_id)
        deployment.status = DeploymentStatus.TERMINATED
        return deployment

    def list_deployments(self) -> Dict[str, Deployment]:
        return dict(self._deployments)
