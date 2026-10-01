"""Multi-cloud provider abstraction (AWS, GCP, Azure, OCI, Alibaba).

Each provider exposes a small, uniform interface for the operations the
Aegis control plane needs (listing compute instances, fetching cost usage,
and basic resource tagging). SDKs are imported lazily.
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class CloudProvider(abc.ABC):
    name: str = "cloud"

    @abc.abstractmethod
    def list_instances(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_cost_and_usage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        raise NotImplementedError

    def is_available(self) -> bool:
        return True


class AWSProvider(CloudProvider):
    name = "aws"

    def __init__(self, region_name: Optional[str] = None):
        self.region_name = region_name

    def is_available(self) -> bool:
        try:
            import boto3  # noqa: F401
        except ImportError:
            return False
        return True

    def list_instances(self) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("boto3 is not installed")
        import boto3

        client = boto3.client("ec2", region_name=self.region_name)
        reservations = client.describe_instances().get("Reservations", [])
        return [instance for r in reservations for instance in r.get("Instances", [])]

    def get_cost_and_usage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("boto3 is not installed")
        import boto3

        client = boto3.client("ce", region_name=self.region_name or "us-east-1")
        return client.get_cost_and_usage(
            TimePeriod={"Start": start_date, "End": end_date},
            Granularity="DAILY",
            Metrics=["UnblendedCost"],
        )


class GCPProvider(CloudProvider):
    name = "gcp"

    def __init__(self, project: str):
        self.project = project

    def is_available(self) -> bool:
        try:
            from google.cloud import compute_v1  # noqa: F401
        except ImportError:
            return False
        return True

    def list_instances(self) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("google-cloud-compute is not installed")
        from google.cloud import compute_v1

        client = compute_v1.InstancesClient()
        aggregated = client.aggregated_list(project=self.project)
        instances = []
        for _, scoped_list in aggregated:
            for instance in getattr(scoped_list, "instances", []):
                instances.append({"name": instance.name, "status": instance.status})
        return instances

    def get_cost_and_usage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "GCP cost reporting requires BigQuery billing export configuration"
        )


class AzureProvider(CloudProvider):
    name = "azure"

    def __init__(self, subscription_id: str):
        self.subscription_id = subscription_id

    def is_available(self) -> bool:
        try:
            from azure.mgmt.compute import ComputeManagementClient  # noqa: F401
        except ImportError:
            return False
        return True

    def list_instances(self) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("azure-mgmt-compute is not installed")
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.compute import ComputeManagementClient

        client = ComputeManagementClient(DefaultAzureCredential(), self.subscription_id)
        return [{"name": vm.name, "location": vm.location} for vm in client.virtual_machines.list_all()]

    def get_cost_and_usage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        raise NotImplementedError("Azure cost reporting requires Cost Management API integration")


class OCIProvider(CloudProvider):
    name = "oci"

    def __init__(self, compartment_id: str):
        self.compartment_id = compartment_id

    def is_available(self) -> bool:
        try:
            import oci  # noqa: F401
        except ImportError:
            return False
        return True

    def list_instances(self) -> List[Dict[str, Any]]:
        if not self.is_available():
            raise RuntimeError("oci SDK is not installed")
        import oci

        config = oci.config.from_file()
        client = oci.core.ComputeClient(config)
        instances = client.list_instances(self.compartment_id).data
        return [{"id": i.id, "lifecycle_state": i.lifecycle_state} for i in instances]

    def get_cost_and_usage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        raise NotImplementedError("OCI cost reporting requires Usage API integration")


class AlibabaProvider(CloudProvider):
    name = "alibaba"

    def __init__(self, region_id: str):
        self.region_id = region_id

    def is_available(self) -> bool:
        try:
            from aliyunsdkcore.client import AcsClient  # noqa: F401
        except ImportError:
            return False
        return True

    def list_instances(self) -> List[Dict[str, Any]]:
        raise NotImplementedError("Alibaba Cloud ECS listing requires aliyun-sdk configuration")

    def get_cost_and_usage(self, start_date: str, end_date: str) -> Dict[str, Any]:
        raise NotImplementedError("Alibaba Cloud cost reporting requires BSS OpenAPI integration")
