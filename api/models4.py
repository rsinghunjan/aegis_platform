  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
 20
 21
 22
 23
 24
 25
 26
 27
 28
 29
 30
 31
 32
 33
 34
 35
 36
 37
 38
 39
 40
 41
 42
 43
 44
 45
 46
 47
 48
 49
 50
# (append) Add tenant, tenant quota, usage and invoice models to existing api/models.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, Float
from sqlalchemy.orm import relationship
from .db import Base

# Tenant model (nullable in single-tenant deployments)
class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(String(100), primary_key=True)  # e.g., UUID or tenant slug
    name = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class TenantQuota(Base):
    __tablename__ = "tenant_quotas"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(100), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)
    rate_per_min = Column(Integer, nullable=True)  # requests per minute
    burst = Column(Integer, nullable=True)  # burst capacity
    daily_quota_units = Column(Integer, nullable=True)  # optional daily quota in units
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    tenant = relationship("Tenant")


class UsageRecord(Base):
    __tablename__ = "usage_records"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(100), nullable=True, index=True)
    model_name = Column(String(200), nullable=False)
    version = Column(String(100), nullable=False)
    units = Column(Integer, nullable=False, default=1)
    inference_ms = Column(Float, nullable=True)
    cost_estimate = Column(Float, nullable=True)
    extra = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Invoice(Base):
    __tablename__ = "invoices"
    id = Column(Integer, primary_key=True)
    tenant_id = Column(String(100), nullable=True, index=True)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    units = Column(Integer, nullable=False, default=0)
    amount = Column(Float, nullable=False, default=0.0)
    currency = Column(String(10), nullable=False, default="USD")
    status = Column(String(50), nullable=False, default="issued")  # issued | paid | disputed
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
