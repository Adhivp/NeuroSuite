from pydantic import BaseModel, EmailStr, Field, validator
import re


class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    organization: str | None = None
    role: str | None = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)
    confirm_password: str
    agreed_to_terms: bool = Field(...)

    @validator('agreed_to_terms')
    def must_agree_to_terms(cls, v):
        if not v:
            raise ValueError('User must agree to terms and privacy policy')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

    @validator('password')
    def password_strength(cls, v):
        # Check for minimum length
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        # Check for at least one digit
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v


class UserResponse(UserBase):
    id: int

    class Config:
        from_attributes = True  # Updated from orm_mode = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class TokenData(BaseModel):
    email: str | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str