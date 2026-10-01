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
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
117
118
119
120
121
122
123
124
125
126
127
128
129
130
131
132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
"""
def _create_refresh_token() -> str:
    # opaque UUID stored in DB
    return str(uuid.uuid4())


def _decode_token(token: str) -> Dict[str, Any]:
    secret = get_jwt_secret()
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid access token") from exc


# ----- helper DB functions -------------------------------------------------


def get_user_by_username(session, username: str) -> Optional[DBUser]:
    return session.query(DBUser).filter_by(username=username).one_or_none()


def create_refresh_token_db(session, user: DBUser, days_valid: int = REFRESH_EXPIRE_DAYS) -> DBRefreshToken:
    token = _create_refresh_token()
    expires = datetime.utcnow() + timedelta(days=days_valid)
    rt = DBRefreshToken(user_id=user.id, token=token, revoked=False, expires_at=expires)
    session.add(rt)
    session.commit()
    session.refresh(rt)
    return rt


def revoke_refresh_token_db(session, token_str: str) -> bool:
    rt = session.query(DBRefreshToken).filter_by(token=token_str).one_or_none()
    if not rt:
        return False
    rt.revoked = True
    session.commit()
    return True


# ----- endpoints ----------------------------------------------------------


@router.post("/token", response_model=Token)
def token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Token endpoint using Resource Owner Password Credentials (for trusted clients).
    production: prefer Authorization Code / PKCE flows with OIDC provider.
    """
    username = form_data.username
    password = form_data.password
    session = SessionLocal()
    try:
        user = get_user_by_username(session, username)
        if not user or not verify_password(password, user.password_hash):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        if user.disabled:
            raise HTTPException(status_code=400, detail="User disabled")
        # build tokens
        scopes = list(user.scopes or [])
        access_token = _create_access_token(sub=user.username, scopes=scopes, tenant=getattr(user, "tenant", None))
        rt = create_refresh_token_db(session, user)
        return Token(access_token=access_token, expires_in=ACCESS_EXPIRE_MINUTES * 60, refresh_token=rt.token, scope=" ".join(scopes))
    finally:
        session.close()


class RefreshIn(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=Token)
def refresh_token(payload: RefreshIn):
    session = SessionLocal()
    try:
