"""
Simple auth server & middleware for the Aegis API (development reference)

Features:
- /auth/token : username/password -> access + refresh tokens
- /auth/refresh : refresh_token -> new access token
- JWT tokens (HS256) with scopes claim
- require_scopes(required_scopes) dependency generator to enforce scopes on endpoints

Production notes (replace before prod):
- Replace in-memory user DB with real identity provider (OAuth2, Keycloak, Cognito, etc).
- Use strong password hashing (bcrypt/argon2) and a secure secrets store for SECRET_KEY.
- Add token revocation storage for refresh tokens.
- Use HTTPS and rotate the SECRET_KEY via a secure mechanism.
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer, SecurityScopes
from pydantic import BaseModel
import jwt
from jwt import InvalidTokenError as JWTError

# Configuration (override via env in production)
SECRET_KEY = os.environ.get("AEGIS_SECRET_KEY", "dev-secret-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("AEGIS_ACCESS_EXPIRE_MINS", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("AEGIS_REFRESH_EXPIRE_DAYS", "7"))

# OAuth2 scheme for dependencies (tokenUrl is the token issuance endpoint)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", scopes={
    "predict": "Run model predictions",
    "model:read": "Read model metadata",
    "admin": "Admin operations"
})

router = APIRouter(prefix="/auth", tags=["auth"])


# ----- Pydantic models --------------------------------------------------------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = ""


class TokenPayload(BaseModel):
    sub: str
    scopes: List[str] = []
    exp: int


class User(BaseModel):
    username: str
    scopes: List[str] = []
    disabled: bool = False


# ----- Fake user DB (DEV only) -----------------------------------------------
# Replace by your user management or external identity provider

_fake_users = {
    "alice": {
        "username": "alice",
        "password": "wonderland",     # DEV: plaintext password. Replace with hashed check.
        "scopes": ["predict", "model:read"]
    },
    "bob": {
        "username": "bob",
        "password": "builder",
        "scopes": ["model:read"]
    },
    "admin": {
        "username": "admin",
        "password": "adminpass",
        "scopes": ["predict", "model:read", "admin"]
    }
}


def get_user(username: str) -> Optional[User]:
    entry = _fake_users.get(username)
    if not entry:
        return None
    return User(username=entry["username"], scopes=entry.get("scopes", []))


def verify_password(plain_password: str, stored_password: str) -> bool:
    # DEV placeholder. In production, use passlib bcrypt/argon2 verify.
    return plain_password == stored_password


# ----- Token helpers ---------------------------------------------------------

def create_access_token(subject: str, scopes: List[str], expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode = {"sub": subject, "scopes": scopes, "exp": int(expire.timestamp())}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode = {"sub": subject, "scope": "refresh", "exp": int(expire.timestamp())}
    encoded = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded


def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


# ----- Endpoints: token issuance & refresh -----------------------------------

@router.post("/token", response_model=Token)
async def token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Token endpoint supporting resource-owner password credentials (dev).
    - grant_type=password : returns access+refresh (scopes from user)
    Note: For production, use an external OAuth2 provider (authorization code / device code flows).
    """
    # Only 'password' grant type supported in this simple implementation
    username = form_data.username
    password = form_data.password
    user_entry = _fake_users.get(username)
    if not user_entry or not verify_password(password, user_entry["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    user = get_user(username)
    if user.disabled:
        raise HTTPException(status_code=400, detail="User disabled")

    access_token = create_access_token(subject=user.username, scopes=user.scopes)
    refresh_token = create_refresh_token(subject=user.username)
    return Token(access_token=access_token, expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                 refresh_token=refresh_token, scope=" ".join(user.scopes))


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=Token)
async def refresh_token(req: RefreshRequest):
    payload = None
    try:
        payload = jwt.decode(req.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = get_user(sub)
    if not user or user.disabled:
        raise HTTPException(status_code=401, detail="Invalid user")
    access_token = create_access_token(subject=user.username, scopes=user.scopes)
    return Token(access_token=access_token, expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60, refresh_token=req.refresh_token,
                 scope=" ".join(user.scopes))


# ----- Dependency factory to enforce scopes ----------------------------------

def require_scopes(required: List[str]):
    """
    Returns a dependency function which:
    - extracts JWT from Authorization header via oauth2_scheme
    - validates token and ensures required scopes are present
    - returns the authenticated User object (pydantic)
    Usage:
        current_user = Depends(require_scopes(["predict"]))
    """
    async def _dependency(security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)) -> User:
        payload = None
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

        token_scopes = payload.get("scopes", [])
        # allow tokens with 'admin' scope to bypass checks
        if "admin" in token_scopes:
            user = get_user(username)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            return user

        missing = [s for s in required if s not in token_scopes]
        if missing:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"Missing required scopes: {missing}")

        user = get_user(username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        if user.disabled:
            raise HTTPException(status_code=400, detail="User disabled")
        return user

    return _dependency
