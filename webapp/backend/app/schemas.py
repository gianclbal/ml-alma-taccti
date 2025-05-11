# app/schemas.py

from pydantic import BaseModel

# UserSignup model for sign-up request
class UserSignup(BaseModel):
    email: str
    password: str

# UserLogin model for login request
class UserLogin(BaseModel):
    email: str
    password: str

# PasswordResetRequest model for requesting a password reset
class PasswordResetRequest(BaseModel):
    email: str

# PasswordResetConfirm model for confirming password reset
class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str