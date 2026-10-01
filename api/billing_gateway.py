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
 51
 52
 53
 54
 55
 56
 57
 58
 59
 60
 61
 62
 63
 64
 65
 66
 67
 68
 69
 70
 71
 72
 73
 74
 75
 76
 77
 78
 79
 80
 81
 82
 83
 84
 85
 86
 87
 88
 89
 90
 91
 92
 93
 94
 95
 96
 97
 98
 99
"""
Payment gateway adapter for Aegis.

- Default implementation uses Stripe (stripe package).
- In dev if STRIPE_API_KEY is not provided the adapter falls back to a simple mock that marks payments as succeeded.
- Exposes:
    create_customer(tenant_id, email, metadata)
    charge_customer(customer_id, amount_cents, currency, description, invoice_id)
    verify_webhook(payload, sig_header)
"""
import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("aegis.billing_gateway")

STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")

try:
    if STRIPE_API_KEY:
        import stripe  # type: ignore
        stripe.api_key = STRIPE_API_KEY
        STRIPE_AVAILABLE = True
    else:
        STRIPE_AVAILABLE = False
except Exception:
    STRIPE_AVAILABLE = False

# ------- Public wrapper functions -------


def create_customer(tenant_id: str, email: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Create or return a payment customer for tenant. Returns minimal dict:
      { "id": "<gateway_customer_id>", "raw": <provider response dict> }
    """
    if STRIPE_AVAILABLE:
        try:
            cust = stripe.Customer.create(email=email, metadata={"tenant_id": tenant_id, **(metadata or {})})
            return {"id": cust.id, "raw": cust}
        except Exception:
            logger.exception("stripe create_customer failed")
            raise
    # fallback: simple mock
    mock_id = f"mock-cust-{tenant_id}"
    logger.info("Using mock payment customer %s (STRIPE not configured)", mock_id)
    return {"id": mock_id, "raw": {"id": mock_id}}


def charge_customer(customer_id: str, amount_cents: int, currency: str = "USD", description: str = "", invoice_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Charge a customer. Returns { success: bool, provider_id: str?, raw: {} , error: str? }.
    Amount is in cents.
    """
    if STRIPE_AVAILABLE:
        try:
            # For simplicity we create a PaymentIntent and confirm it immediately (requires saved payment method in Stripe)
            intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency=currency.lower(),
                customer=customer_id,
                off_session=True,
                confirm=True,
                description=description,
                metadata={"aegis_invoice_id": str(invoice_id) if invoice_id else ""}
            )
            status = intent.status
            provider_id = getattr(intent, "id", None)
            success = status in ("succeeded",)
            return {"success": success, "provider_id": provider_id, "raw": intent}
        except stripe.error.CardError as ce:
            logger.warning("Card error charging customer %s: %s", customer_id, ce)
            return {"success": False, "error": str(ce)}
        except Exception:
            logger.exception("stripe charge failed")
            return {"success": False, "error": "gateway_error"}
    # fallback: always succeed in dev
    logger.info("Mock charge of %d %s for customer %s (invoice=%s)", amount_cents, currency, customer_id, invoice_id)
    return {"success": True, "provider_id": f"mock-charge-{invoice_id}", "raw": {}}


def verify_webhook(payload: bytes, sig_header: str) -> Optional[Dict[str, Any]]:
    """
    Verify a Stripe webhook signature and return the parsed event, or None if not verifiable.
    """
    if STRIPE_AVAILABLE and STRIPE_WEBHOOK_SECRET:
        try:
            event = stripe.Webhook.construct_event(payload=payload, sig_header=sig_header, secret=STRIPE_WEBHOOK_SECRET)
            return event
        except Exception:
            logger.exception("stripe webhook verify failed")
            return None
    # fallback: no verification, attempt to parse JSON if payload is bytes
    try:
        import json
        return json.loads(payload.decode("utf-8"))
    except Exception:
        return None
api/billing_gateway.py
