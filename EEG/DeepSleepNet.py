import numpy as np
import torch
from braindecode.models import DeepSleepNet
from braindecode.preprocessing import exponential_moving_standardize
import mne
import os

def load_pretrained_model(model_path=None):
    """
    Load a pretrained DeepSleepNet model
    
    Parameters:
    -----------
    model_path : str or None
        Path to pretrained model weights, if None will use a new model
        
    Returns:
    --------
    model : braindecode.models.DeepSleepNet
        DeepSleepNet model for sleep stage classification
    """
    # Create DeepSleepNet model
    model = DeepSleepNet(
        n_outputs=5,  # 5 sleep stages
        n_chans=1     # Single EEG channel
    )
    
    # Load pretrained weights if provided
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, 
                                            map_location=torch.device('cpu')))
            print(f"Loaded pretrained model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Using untrained model. Predictions may not be accurate.")
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def preprocess_eeg_data(eeg_data, sfreq=100, target_length=30):
    """
    Preprocess EEG data for DeepSleepNet
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (channels, time_points)
    sfreq : float
        Sampling frequency of the EEG data
    target_length : int
        Length of window in seconds (default: 30s for sleep staging)
        
    Returns:
    --------
    processed_data : numpy.ndarray
        Preprocessed EEG data ready for model input
    """
    # Ensure data is 2D (channels, time_points)
    if eeg_data.ndim == 1:
        eeg_data = eeg_data.reshape(1, -1)
    
    # Resample if needed to 100 Hz (expected by model)
    target_sfreq = 100
    if sfreq != target_sfreq:
        print(f"Resampling from {sfreq}Hz to {target_sfreq}Hz")
        n_samples = int((eeg_data.shape[1] / sfreq) * target_sfreq)
        eeg_data = mne.filter.resample(eeg_data, up=target_sfreq, down=sfreq, n_jobs=1)
    
    # Check if data length matches target (30s at 100Hz = 3000 samples)
    target_samples = target_length * target_sfreq
    if eeg_data.shape[1] < target_samples:
        # Pad with zeros if too short
        pad_width = ((0, 0), (0, target_samples - eeg_data.shape[1]))
        eeg_data = np.pad(eeg_data, pad_width, mode='constant')
    elif eeg_data.shape[1] > target_samples:
        # Truncate if too long
        eeg_data = eeg_data[:, :target_samples]
    
    # Apply bandpass filter (0.3-30 Hz)
    eeg_data = mne.filter.filter_data(eeg_data, sfreq=target_sfreq, l_freq=0.3, h_freq=30, 
                                     verbose=False, method='fir', fir_design='firwin')
    
    # Standardize the data
    eeg_data = exponential_moving_standardize(eeg_data, factor_new=1e-3, init_block_size=1000)
    
    return eeg_data

def classify_sleep_stage(eeg_data, model=None, model_path=None, sfreq=100):
    """
    Classify sleep stage from EEG data
    
    Parameters:
    -----------
    eeg_data : numpy.ndarray
        EEG data with shape (channels, time_points) or (time_points,)
    model : braindecode.models.DeepSleepNet or None
        Pretrained model or None to load from model_path
    model_path : str or None
        Path to pretrained model weights
    sfreq : float
        Sampling frequency of the EEG data
        
    Returns:
    --------
    stage : int
        Predicted sleep stage (0: Wake, 1: N1, 2: N2, 3: N3/N4, 4: REM)
    stage_name : str
        Name of the predicted sleep stage
    probabilities : numpy.ndarray
        Probabilities for each sleep stage
    """
    # Sleep stage labels
    stage_names = ['Wake', 'N1', 'N2', 'N3/N4', 'REM']
    
    # Load model if not provided
    if model is None:
        model = load_pretrained_model(model_path)
    
    # Preprocess the data
    processed_data = preprocess_eeg_data(eeg_data, sfreq=sfreq)
    
    # Convert to tensor and add batch dimension
    X = torch.FloatTensor(processed_data).unsqueeze(0)
    
    # Perform prediction
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted = torch.argmax(probabilities, dim=1).item()
    
    return predicted, stage_names[predicted], probabilities.cpu().numpy()[0]

def load_eeg_from_file(filepath, channel_idx=0):
    """
    Load EEG data from various file formats using MNE
    
    Parameters:
    -----------
    filepath : str
        Path to the EEG file
    channel_idx : int
        Index of the channel to use for classification
        
    Returns:
    --------
    eeg_data : numpy.ndarray
        EEG data for the selected channel
    sfreq : float
        Sampling frequency of the data
    """
    # Check file extension
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    try:
        # Load data based on file type
        if ext == '.edf':
            raw = mne.io.read_raw_edf(filepath, preload=True)
        elif ext == '.bdf':
            raw = mne.io.read_raw_bdf(filepath, preload=True)
        elif ext in ['.fif', '.fiff']:
            raw = mne.io.read_raw_fif(filepath, preload=True)
        elif ext == '.set':
            raw = mne.io.read_raw_eeglab(filepath, preload=True)
        else:
            # Try to load as generic binary format
            raw = mne.io.read_raw(filepath, preload=True)
        
        # Get data from the selected channel
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        if channel_idx >= data.shape[0]:
            print(f"Warning: Requested channel {channel_idx} out of bounds. Using first channel instead.")
            channel_idx = 0
        
        eeg_data = data[channel_idx]
        
        return eeg_data, sfreq
    
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, None

def example_usage():
    """Example showing how to use the code"""
    # 1. Load pretrained model (replace with path to your model)
    model_path = "deepsleepnet_pretrained.pth"
    model = load_pretrained_model(model_path)
    
    # 2. Option A: Load EEG data from file
    filepath = "sample_eeg.edf"
    if os.path.exists(filepath):
        eeg_data, sfreq = load_eeg_from_file(filepath)
        if eeg_data is not None:
            # Classify sleep stage
            stage, stage_name, probs = classify_sleep_stage(eeg_data, model=model, sfreq=sfreq)
            print(f"Predicted sleep stage: {stage_name} (class {stage})")
            print(f"Probabilities: {probs}")
    
    # 2. Option B: Use numpy array directly
    # This simulates random EEG data for demonstration
    print("\nDemonstration with random data:")
    random_eeg = np.random.randn(3000)  # 30 seconds at 100 Hz
    stage, stage_name, probs = classify_sleep_stage(random_eeg, model=model, sfreq=100)
    print(f"Predicted sleep stage: {stage_name} (class {stage})")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    example_usage()