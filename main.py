from fastapi import FastAPI, Depends, HTTPException, status, Security, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import numpy as np
import sys
from pydantic import BaseModel
import mne

# Add EEG directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

import models
import schemas
import auth
from database import engine, get_db
from EEG.DeepSleepNet import (
    load_pretrained_model, 
    classify_sleep_stage, 
    load_eeg_from_file,
    preprocess_eeg_data
)

# Create the database tables
models.Base.metadata.create_all(bind=engine)

# Load the DeepSleepNet model at application startup
MODEL_PATH = os.environ.get("DEEPSLEEPNET_MODEL_PATH", os.path.join(os.path.dirname(__file__), "EEG/models/deepsleepnet_pretrained.pth"))
try:
    deepsleep_model = load_pretrained_model(MODEL_PATH)
    print(f"DeepSleepNet model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"Warning: Failed to load DeepSleepNet model: {e}")
    print(f"Sleep classification will be inaccurate until a model is loaded")
    deepsleep_model = load_pretrained_model(None)  # Create untrained model

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


# DeepSleepNet Routes

@app.post("/eeg/classify-file", response_model=schemas.SleepClassificationResponse, tags=["Sleep Classification"])
async def classify_eeg_file(
    file: UploadFile = File(...),
    channel_idx: int = Form(0),
    current_user: models.User = Security(auth.get_current_active_user)
):
    """
    Upload an EEG file and classify the sleep stage.
    
    Supports EDF, BDF, FIF, and EEGLAB .set file formats.
    
    Parameters:
    - **file**: EEG data file (required)
    - **channel_idx**: Index of the EEG channel to use (default: 0)
    
    Returns:
    - **stage**: Predicted sleep stage index
    - **stage_name**: Name of the predicted sleep stage
    - **probabilities**: Probability scores for each sleep stage
    """
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)
    
    try:
        # Process the EEG file
        eeg_data, sfreq = load_eeg_from_file(temp_file_path, channel_idx=channel_idx)
        
        if eeg_data is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not load EEG data from file. Make sure it's a supported format."
            )
        
        # Classify sleep stage
        stage, stage_name, probs = classify_sleep_stage(
            eeg_data, 
            model=deepsleep_model, 
            sfreq=sfreq
        )
        
        # Convert numpy array to list for JSON serialization
        probs_list = probs.tolist()
        
        return {
            "stage": stage,
            "stage_name": stage_name,
            "probabilities": probs_list
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing EEG data: {str(e)}"
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/eeg/file-info", response_model=schemas.EEGFileInfo, tags=["General EEG Info"])
async def get_eeg_file_info(
    file: UploadFile = File(...),
    current_user: models.User = Security(auth.get_current_active_user)
):
    """
    Get information about an uploaded EEG file.
    
    Parameters:
    - **file**: EEG data file (required)
    
    Returns:
    - **channels**: List of channel names
    - **sampling_frequency**: Sampling frequency in Hz
    - **duration_seconds**: Duration of the recording in seconds
    - **num_samples**: Number of samples in the recording
    """
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file_path = temp_file.name
        content = await file.read()
        temp_file.write(content)
    
    try:
        # Read the EDF file with MNE to get its properties
        try:
            raw = mne.io.read_raw_edf(temp_file_path, preload=False)
            
            # Extract the required information
            channels = raw.ch_names
            sampling_frequency = raw.info['sfreq']
            duration_seconds = raw.n_times / sampling_frequency
            num_samples = raw.n_times
            
            return {
                "channels": channels,
                "sampling_frequency": float(sampling_frequency),
                "duration_seconds": float(duration_seconds),
                "num_samples": int(num_samples)
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not parse EEG file: {str(e)}"
            )
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.get("/eeg/sleep-stages", response_model=List[schemas.SleepStageInfo], tags=["Sleep Classification"])
def get_sleep_stage_info():
    """
    Get information about the sleep stages classified by DeepSleepNet.
    
    Returns details about each sleep stage and their meanings.
    """
    return [
        {
            "id": 0,
            "name": "Wake", 
            "description": "Awake state, characterized by alpha and beta brain waves. This is when you're conscious and alert."
        },
        {
            "id": 1,
            "name": "N1", 
            "description": "Light sleep, the transition from wakefulness to sleep. Characterized by theta waves and slow eye movements."
        },
        {
            "id": 2, 
            "name": "N2", 
            "description": "Intermediate sleep stage with sleep spindles and K-complexes. Body temperature drops and heart rate slows."
        },
        {
            "id": 3, 
            "name": "N3/N4", 
            "description": "Deep sleep or slow-wave sleep. Characterized by delta waves. Important for physical recovery and memory consolidation."
        },
        {
            "id": 4, 
            "name": "REM", 
            "description": "Rapid Eye Movement sleep. Dream state with brain activity similar to wakefulness. Important for cognitive functions and memory."
        }
    ]


@app.get("/eeg/model-info", tags=["Sleep Classification"])
def get_model_info():
    """
    Get information about the currently loaded DeepSleepNet model.
    
    Returns the model's status and configuration.
    """
    global deepsleep_model
    
    # Check if the model is properly loaded
    model_loaded = deepsleep_model is not None
    
    # Information about the model
    return {
        "model_name": "DeepSleepNet",
        "is_loaded": model_loaded,
        "model_path": MODEL_PATH if model_loaded else None,
        "input_requirements": {
            "sampling_frequency": 100,
            "window_length_seconds": 30,
            "recommended_channels": ["Fpz-Cz", "Pz-Oz", "EEG", "C4-A1"],
            "preprocessing": "Bandpass filtered 0.3-30Hz, standardized"
        },
        "output_classes": [
            {"id": 0, "name": "Wake"},
            {"id": 1, "name": "N1"},
            {"id": 2, "name": "N2"},
            {"id": 3, "name": "N3/N4"},
            {"id": 4, "name": "REM"}
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)