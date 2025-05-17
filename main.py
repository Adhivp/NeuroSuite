from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List
import uvicorn

import models
import schemas
import auth
from database import engine, get_db

# Create the database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="NeuroSuite API",
    description="Backend API for NeuroSuite application",
    version="1.0.0",
)

security = HTTPBearer()

@app.post("/signup", response_model=schemas.UserResponse, tags=["Authentication"])
def signup(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """
    Sign up a new user with the following information:
    - Full Name
    - Email Address
    - Password (with strength validation)
    - Organization (optional)
    - Role / Job Title (optional)
    - Agreement to Terms & Privacy Policy
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
    Login with email and password to receive access and refresh tokens
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
    Get a new access token using a refresh token
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
    Get information about the currently authenticated user
    """
    return current_user


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {"status": "healthy", "service": "NeuroSuite API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)