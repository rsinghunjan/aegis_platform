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
# Append these model definitions to your existing api/models.py (BillingAccount, PaymentMethod)
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime

class BillingAccount(Base):
    __tablename__ = "billing_accounts"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(String(100), nullable=False, index=True)
    gateway_customer_id = Column(String(200), nullable=True)  # e.g., stripe customer id
    billing_contact_email = Column(String(200), nullable=True)
    default_currency = Column(String(10), nullable=False, default="USD")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    payment_methods = relationship("PaymentMethod", back_populates="billing_account")


class PaymentMethod(Base):
    __tablename__ = "payment_methods"
    id = Column(Integer, primary_key=True, index=True)
    billing_account_id = Column(Integer, ForeignKey("billing_accounts.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String(50), nullable=False)  # e.g., "stripe"
    provider_pm_id = Column(String(200), nullable=True)  # e.g., stripe payment method id
    pm_type = Column(String(50), nullable=True)
    last4 = Column(String(10), nullable=True)
    exp_month = Column(Integer, nullable=True)
    exp_year = Column(Integer, nullable=True)
    is_default = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    billing_account = relationship("BillingAccount", back_populates="payment_methods")
