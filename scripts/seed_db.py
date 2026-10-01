"""
Seed the DB with an admin user and a sample model + version.

Usage:
  DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/aegis python scripts/seed_db.py
"""
import os
from datetime import datetime, timedelta
import uuid

from sqlalchemy import select
from passlib.hash import bcrypt
from api.db import SessionLocal, engine
from api.models import Base, User, Model, ModelVersion, RefreshToken

# Ensure tables exist (for dev convenience only)
Base.metadata.create_all(bind=engine)


def create_user(session, username: str, password: str, scopes):
    pw_hash = bcrypt.hash(password)
    user = User(username=username, password_hash=pw_hash, scopes=scopes, disabled=False)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def create_model_and_version(session, name: str, version: str, description: str):
    model = Model(name=name, description=description)
    session.add(model)
    session.commit()
    session.refresh(model)
    mv = ModelVersion(model_id=model.id, version=version, metadata={"description": description}, artifact_path=None)
    session.add(mv)
    session.commit()
    session.refresh(mv)
    return model, mv


def create_refresh_token(session, user, days_valid=7):
    token = str(uuid.uuid4())
    expires = datetime.utcnow() + timedelta(days=days_valid)
    rt = RefreshToken(user_id=user.id, token=token, revoked=False, expires_at=expires)
    session.add(rt)
    session.commit()
    session.refresh(rt)
    return rt


def main():
    session = SessionLocal()
    # admin user
    admin = session.execute(select(User).filter_by(username="admin")).scalar_one_or_none()
    if not admin:
        admin = create_user(session, "admin", "adminpass", ["predict", "model:read", "admin"])
        print("Created admin user: admin / adminpass (change in prod)")

    # example user
    alice = session.execute(select(User).filter_by(username="alice")).scalar_one_or_none()
    if not alice:
        alice = create_user(session, "alice", "wonderland", ["predict", "model:read"])
        print("Created user alice / wonderland")

    # sample model
    m, mv = session.execute(select(Model).filter_by(name="multimodal_demo")).scalar_one_or_none(), None
    if not m:
        m, mv = create_model_and_version(session, "multimodal_demo", "v1", "Demo multimodal model seeded")
        print(f"Created model {m.name} v{mv.version}")

    # refresh tokens for admin (dev convenience)
    rt = session.execute(select(RefreshToken).filter_by(user_id=admin.id)).scalar_one_or_none()
    if not rt:
        rt = create_refresh_token(session, admin)
        print("Admin refresh token:", rt.token)

    session.close()


if __name__ == "__main__":
    main()
