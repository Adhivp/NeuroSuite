from pydantic import BaseModel, EmailStr, Field, validator
import re


class UserBase(BaseModel):
    email: EmailStr = Field(
        ...,
        title="Email",
        description="Valid email address used for account verification and login",
        example="user@example.com"
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        title="Full Name",
        description="User's complete name (2-100 characters)",
        example="John Smith"
    )
    organization: str | None = Field(
        None,
        min_length=2,
        max_length=100,
        title="Organization",
        description="User's company or organization (2-100 characters, optional)",
        example="Acme Corporation"
    )
    role: str | None = Field(
        None,
        min_length=2,
        max_length=100,
        title="Role",
        description="User's job title or role (2-100 characters, optional)",
        example="Software Engineer"
    )


class UserCreate(UserBase):
    password: str = Field(
        ...,
        min_length=8,
        max_length=64,
        title="Password",
        description="Password (8-64 characters) must include at least one uppercase letter, one lowercase letter, one digit, and one special character",
        example="StrongP@ssw0rd"
    )
    confirm_password: str = Field(
        ...,
        min_length=8,
        max_length=64,
        title="Confirm Password",
        description="Must match the password field exactly",
        example="StrongP@ssw0rd"
    )
    agreed_to_terms: bool = Field(
        ...,
        title="Terms Agreement",
        description="User must agree to terms and privacy policy (must be true)",
        example=True
    )

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
    id: int = Field(
        ...,
        title="User ID",
        description="Unique identifier for the user",
        example=1
    )

    class Config:
        from_attributes = True  # Updated from orm_mode = True


class Token(BaseModel):
    access_token: str = Field(
        ...,
        title="Access Token",
        description="JWT token for API authentication (valid for limited time)",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNjUwMDAwMDAwfQ.signature"
    )
    refresh_token: str = Field(
        ...,
        title="Refresh Token",
        description="JWT token used to obtain a new access token when it expires",
        example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNjUwMDAwMDAwLCJ0eXBlIjoicmVmcmVzaCJ9.signature"
    )
    token_type: str = Field(
        ...,
        title="Token Type",
        description="Type of authentication token (always 'bearer')",
        example="bearer"
    )


class TokenData(BaseModel):
    email: str | None = Field(
        None,
        title="Email",
        description="Email address extracted from the JWT token",
        example="user@example.com"
    )


class UserLogin(BaseModel):
    email: EmailStr = Field(
        ...,
        title="Email",
        description="Registered email address",
        example="user@example.com"
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=64,
        title="Password",
        description="User password (8-64 characters)",
        example="StrongP@ssw0rd"
    )