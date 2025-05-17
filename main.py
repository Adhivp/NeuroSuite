from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

import models
import schemas
import auth
from database import engine, get_db

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="NeuroSuite API",
    description="""
    Backend API for NeuroSuite application.
    """,
    version="1.0.0",
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

security = HTTPBearer()

@app.post("/signup", response_model=schemas.UserResponse, tags=["Authentication"])
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Sign up a new user with the following requirements:
    
    - **Email**: Valid unique email address (required)
    - **Full Name**: 2-100 characters (required)
    - **Password**: 8-64 characters with complexity requirements (required):
      - At least one uppercase letter
      - At least one lowercase letter
      - At least one digit
      - At least one special character
    - **Confirm Password**: Must match password exactly
    - **Organization**: 2-100 characters (optional)
    - **Role**: 2-100 characters (optional)
    - **Terms Agreement**: Must be accepted (true)
    
    Returns the created user without sensitive information.
    """
    # Check if user with this email already exists
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password,
        organization=user.organization,
        role=user.role
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post("/login", response_model=schemas.Token, tags=["Authentication"])
def login(user_credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    """
    Login with email and password to receive access and refresh tokens.
    
    - **Email**: Registered email address (required)
    - **Password**: User's password (required)
    
    Returns:
    - **access_token**: JWT token for API authentication
    - **refresh_token**: JWT token used to obtain a new access token
    - **token_type**: Type of authentication token (bearer)
    """
    user = auth.authenticate_user(db, user_credentials.email, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token, refresh_token = auth.create_tokens({"sub": user.email})
    
    return {
        "access_token": access_token, 
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@app.post("/refresh-token", response_model=schemas.Token, tags=["Authentication"])
def refresh_token(credentials: HTTPAuthorizationCredentials = Security(security), db: Session = Depends(get_db)):
    """
    Get a new access token using a refresh token.
    
    Provide your refresh token in the Authorization header with the Bearer prefix.
    
    Returns:
    - **access_token**: New JWT token for API authentication
    - **refresh_token**: New JWT token for future refresh operations
    - **token_type**: Type of authentication token (bearer)
    """
    refresh_token = credentials.credentials
    payload = auth.get_token_payload(refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    email = payload.get("sub")
    user = auth.get_user(db, email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token, new_refresh_token = auth.create_tokens({"sub": user.email})
    
    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer"
    }


@app.get("/users/me", response_model=schemas.UserResponse, tags=["Users"])
def read_users_me(current_user: models.User = Security(auth.get_current_active_user)):
    """
    Get information about the currently authenticated user.
    
    Requires a valid access token in the Authorization header.
    
    Returns the authenticated user's profile information:
    - **id**: User ID
    - **email**: Email address
    - **full_name**: User's full name
    - **organization**: User's organization (if provided)
    - **role**: User's job title or role (if provided)
    """
    return current_user


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify API is running.
    
    This endpoint does not require authentication.
    
    Returns a simple status message confirming the API is operational.
    """
    return {"status": "healthy", "service": "NeuroSuite API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)