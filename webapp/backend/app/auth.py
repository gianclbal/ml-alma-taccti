from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import uuid

router = APIRouter()

# Secret and hashing config
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
PASSWORD_RESET_EXPIRE_MINUTES = 10  # Password reset token expiration time

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
fake_user_db = {}  # To simulate database (replace with actual DB later)
password_reset_tokens = {}  # To temporarily store password reset tokens

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Define Pydantic models for request bodies
class UserSignup(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

# Utility functions for password hashing and verification
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Function to extract the current user from the JWT token
def get_current_user(token: str = Depends(api_key_header)):
    # if not token:
    #     raise HTTPException(status_code=401, detail="Authorization token is required.")
        
    if not token:
        return "test_user"  # Allow default user for testing when no token is provided
    
    if not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = token[7:]  # Remove "Bearer " prefix
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Endpoint for user sign-up
@router.post("/signup")
def signup(user: UserSignup):
    print(f"Received signup request: {user}")
    if user.email in fake_user_db:
        raise HTTPException(status_code=400, detail="Email already exists.")
    hashed_password = get_password_hash(user.password)
    fake_user_db[user.email] = {"email": user.email, "hashed_password": hashed_password}
    return {"message": f"User {user.email} signed up successfully!"}

# Endpoint for user login
@router.post("/login")
def login(user: UserLogin):
    db_user = fake_user_db.get(user.email)
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    access_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint for password reset request (user submits email)
@router.post("/reset-password")
def reset_password(request: PasswordResetRequest):
    if request.email not in fake_user_db:
        raise HTTPException(status_code=404, detail="User not found")

    # Generate a password reset token and expiration time
    token = str(uuid.uuid4())
    expiration_time = datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_EXPIRE_MINUTES)

    # Store the reset token and expiration time
    password_reset_tokens[token] = {"email": request.email, "expires_at": expiration_time}

    # In a real-world scenario, you would send this token to the user's email.
    print(f"Password reset token (for testing): {token}")  # This would be an email in production

    return {"message": "Password reset link has been sent to your email (simulated).", "token": token}

# Endpoint to confirm password reset and update password
@router.post("/reset-password-confirm")
def reset_password_confirm(request: PasswordResetConfirm):
    # Check if token exists and is valid
    token_data = password_reset_tokens.get(request.token)
    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    # Check if the token has expired
    if token_data["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Token has expired")

    # Update the user's password in the fake database
    email = token_data["email"]
    if email not in fake_user_db:
        raise HTTPException(status_code=404, detail="User not found")
    fake_user_db[email]["hashed_password"] = get_password_hash(request.new_password)

    # Remove the token after use
    del password_reset_tokens[request.token]

    return {"message": "Password reset successful."}