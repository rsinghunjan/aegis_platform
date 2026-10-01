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
# (append) Add TriageItem model to existing api/models.py
"""
Append-only patch for triage items used by safety/human review.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from .db import Base

class TriageItem(Base):
    __tablename__ = "triage_items"
    id = Column(Integer, primary_key=True, index=True)
    safety_event_id = Column(Integer, ForeignKey("safety_events.id", ondelete="SET NULL"), nullable=True, index=True)
    request_id = Column(String(100), nullable=True, index=True)
    status = Column(String(20), nullable=False, default="open")  # open|in_progress|resolved
    assigned_to = Column(String(200), nullable=True)
    reasons = Column(JSON, nullable=True)
    input_snapshot = Column(Text, nullable=True)
    review_note = Column(Text, nullable=True)
    resolution = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    resolved_at = Column(DateTime, nullable=True)

    # relationship back to SafetyEvent (optional)
    safety_event = relationship("SafetyEvent", backref="triage_items")
