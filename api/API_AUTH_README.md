# API Authentication (DEV reference)

Files:
- api/auth.py       (JWT token issuance and scope enforcement)
- api/api_server.py (updated to include auth router and scope checks)

Dev quickstart:
1. Install deps:
   pip install -r api/requirements-api.txt

2. Start the API:
   python -m api.api_server

3. Get a token (dev user 'alice' exists in the demo DB):
   curl -X POST "http://localhost:8080/auth/token" -F "username=alice" -F "password=wonderland"

   Response contains access_token and refresh_token.

4. Use token for prediction:
   curl -X POST "http://localhost:8080/v1/models/multimodal_demo/versions/v1/predict" \
     -H "Authorization: Bearer <ACCESS_TOKEN>" \
     -H "Content-Type: application/json" \
     -d '{"text":"hello"}'

Notes & next steps:
- The demo uses a simple in-memory user DB and plaintext passwords. Replace with a real user store (DB) and hashed passwords.
- For production, delegate authentication to a proper OAuth2 provider (Auth0, Keycloak, Cognito) or implement an Authorization Code flow instead of password grant.
- Add refresh token revocation and rotation.
- Integrate roles/scopes with your tenant/organization model for multi-tenant authorization.
