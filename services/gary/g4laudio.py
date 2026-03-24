import os
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

# --- begin Torch/Transformers compat shim (for PyTorch 2.1.x) ---
try:
    import torch
    # only import if torch is present (it is), then alias _register→register for 2.1.x
    from torch.utils import _pytree as _torch_pytree  # type: ignore
    if (hasattr(_torch_pytree, "_register_pytree_node")
            and not hasattr(_torch_pytree, "register_pytree_node")):
        # mimic the newer API expected by recent transformers
        _torch_pytree.register_pytree_node = _torch_pytree._register_pytree_node  # type: ignore[attr-defined]
except Exception:
    # don't hard-crash if anything is odd; worst case we fall back to the original error
    pass
import torchaudio
import torch
import numpy as np
from audiocraft.models import MusicGen
import base64
import io
import uuid
import torchaudio.transforms as T
import gc
from typing import Optional, Callable, List, Any, Union, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from threading import Lock
from functools import wraps

# Add to g4laudio.py after imports
from weakref import WeakValueDictionary
import hashlib
import time

class ModelKernelCache:
    """Cache compiled CUDA kernels across model instances."""
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelKernelCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._kernel_cache = {}
        self._initialized = True
    
    def get_cache_key(self, model_name, device_id):
        """Generate a cache key for model+device combination."""
        key_str = f"{model_name}_{device_id}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def has_warmed_kernels(self, model_name, device_id):
        """Check if we've already warmed up kernels for this model."""
        key = self.get_cache_key(model_name, device_id)
        return key in self._kernel_cache
    
    def mark_kernels_warmed(self, model_name, device_id):
        """Mark kernels as warmed for this model."""
        key = self.get_cache_key(model_name, device_id)
        self._kernel_cache[key] = {
            'warmed_at': time.time(),
            'model_name': model_name,
            'device_id': device_id
        }
        print(f"[KERNEL] Marked kernels as warmed for {model_name} on device {device_id}")

# Global instance
kernel_cache = ModelKernelCache()

def aggressive_cpu_cleanup():
    """More thorough cleanup of CPU memory."""
    import ctypes
    
    # Force garbage collection
    for _ in range(3):
        gc.collect()
    
    # Release memory back to OS (Linux)
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def configure_cuda_for_stability():
    """
    Configure CUDA settings to prevent dtype issues under heavy concurrent load.
    
    This function applies multiple stability-focused settings:
    1. Disables TF32 automatic optimization
    2. Sets deterministic algorithms where possible
    3. Controls memory usage and fragmentation
    4. Disables automatic mixed precision optimizations
    """
    if torch.cuda.is_available():
        # Disable TF32 to force full float32 precision
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Force deterministic algorithms
        torch.backends.cudnn.deterministic = True
        
        # Set a reasonable benchmark setting
        torch.backends.cudnn.benchmark = False
        
        # Control memory usage to prevent fragmentation
        # Only set this if you're using PyTorch 1.10+ 
        if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, 'set_per_process_memory_fraction'):
            torch.cuda.memory.set_per_process_memory_fraction(0.85)
        
        # Force consistent device usage
        torch.cuda.set_device(0)  # Update this if using multiple GPUs
        
        # Empty cache initially
        torch.cuda.empty_cache()
        gc.collect()
        
        print("CUDA configured for stability and consistent dtype handling")
    else:
        print("CUDA not available, no configuration applied")

@contextmanager
def enforced_dtype_context():
    """Context manager that enforces float32 computation for the duration of its scope."""
    # Save original default dtype
    original_dtype = torch.get_default_dtype()
    
    # Force float32 as default dtype
    torch.set_default_dtype(torch.float32)
    
    try:
        yield
    finally:
        # Restore original default dtype
        torch.set_default_dtype(original_dtype)
        
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@contextmanager
def cuda_execution_context(device_id=0):
    """
    Simplified CUDA context - manages device and dtype only.
    No forced synchronization - let PyTorch handle async operations.
    """
    orig_default_dtype = torch.get_default_dtype()
    
    try:
        torch.set_default_dtype(torch.float32)
        
        if torch.cuda.is_available():
            with torch.cuda.device(device_id):
                yield
        else:
            yield
    finally:
        torch.set_default_dtype(orig_default_dtype)
        # Lightweight cleanup only
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@contextmanager
def tensor_lifecycle():
    """
    Context manager for tracking and cleaning up tensors.
    Usage:
        with tensor_lifecycle() as track_tensor:
            tensor1 = some_operation()
            track_tensor(tensor1)
            # More operations...
    """
    tensors: List[torch.Tensor] = []
    
    def track_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is not None and isinstance(tensor, torch.Tensor):
            tensors.append(tensor)
        return tensor
    
    try:
        yield track_tensor
    finally:
        for tensor in tensors:
            try:
                # Check if tensor still exists and has not been freed
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    del tensor
            except:
                pass
        tensors.clear()
        torch.cuda.empty_cache()

class ResamplerPool:
    """
    Singleton pool for managing resampler instances.
    Reuses existing resamplers for common sample rate conversions.
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResamplerPool, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._resamplers = {}
        self._device = None
        self._initialized = True
    
    def get_resampler(self, orig_sr: int, target_sr: int, device: Optional[str] = None) -> T.Resample:
        """
        Get or create a resampler for the specified sample rates.
        
        Args:
            orig_sr: Original sample rate
            target_sr: Target sample rate
            device: Device to place resampler on (optional)
        
        Returns:
            T.Resample: Resampler instance
        """
        if device and device != self._device:
            self.clear()  # Clear pool if device changes
            self._device = device
            
        key = (orig_sr, target_sr)
        if key not in self._resamplers:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
            if device:
                resampler = resampler.to(device)
            self._resamplers[key] = resampler
        return self._resamplers[key]
    
    def clear(self):
        """Clear all resamplers from the pool and free memory."""
        self._resamplers.clear()
        torch.cuda.empty_cache()

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors with detailed messages."""
    pass

@dataclass
class AudioConfig:
    """Configuration for audio processing parameters."""
    prompt_duration: float = 6.0
    top_k: int = 250
    temperature: float = 1.0
    cfg_coef: float = 3.0
    target_sr: int = 32000
    output_duration: float = 30.0

@contextmanager
def resource_cleanup():
    """Lightweight resource cleanup - no forced synchronization."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Add this helper at the top of the file
def get_device():
    """Get the most appropriate device available."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

# Audio preprocessing utility functions
def peak_normalize(y, target_peak=0.9):
    """Normalize audio to a target peak amplitude."""
    return target_peak * (y / np.max(np.abs(y)))

def rms_normalize(y, target_rms=0.05):
    """Normalize audio to a target RMS value."""
    current_rms = np.sqrt(np.mean(y**2))
    return y * (target_rms / current_rms)

def preprocess_audio(waveform):
    """Preprocess audio waveform while preserving proper tensor dimensions."""
    device = get_device()
    
    # Store original shape for reference
    original_shape = waveform.shape
    print(f"🔍 preprocess_audio input shape: {original_shape}")
    
    # Handle different input shapes properly
    if waveform.dim() == 3:
        # [B, C, T] format - extract the audio data
        waveform_np = waveform.cpu().squeeze(0).squeeze(0).numpy()  # Remove batch and channel dims -> [T]
        target_shape = (1, 1, -1)  # Will reshape back to [1, 1, T]
    elif waveform.dim() == 2:
        # [C, T] format - extract the audio data  
        waveform_np = waveform.cpu().squeeze(0).numpy()  # Remove channel dim -> [T]
        target_shape = (1, 1, -1)  # Will reshape back to [1, 1, T]
    elif waveform.dim() == 1:
        # [T] format - already the right dimension
        waveform_np = waveform.cpu().numpy()
        target_shape = (1, 1, -1)  # Will reshape to [1, 1, T]
    else:
        raise ValueError(f"Unsupported waveform dimensions: {original_shape}")
    
    # Do any processing on the 1D numpy array
    processed_waveform_np = waveform_np  # Currently no processing, but could add normalization etc.
    
    # Convert back to tensor with proper 3D shape [1, 1, T]
    result = torch.from_numpy(processed_waveform_np).reshape(target_shape).to(device)
    
    print(f"🔍 preprocess_audio output shape: {result.shape}")
    
    return result

def wrap_audio_if_needed(waveform, sr, desired_duration):
    """Wrap audio if needed to match desired duration."""
    current_duration = waveform.shape[-1] / sr
    
    print(f"🔍 wrap_audio_if_needed input: {waveform.shape}, duration: {current_duration:.2f}s, target: {desired_duration:.2f}s")
    
    # If the current duration is already longer than or equal to the desired duration, return as is
    if current_duration >= desired_duration:
        print(f"✅ No padding needed (already {current_duration:.2f}s >= {desired_duration:.2f}s)")
        return waveform

    # Calculate how much silence is needed (in samples)
    padding_duration = desired_duration - current_duration
    padding_samples = int(padding_duration * sr)
    
    print(f"🔄 Adding {padding_duration:.2f}s ({padding_samples} samples) of padding")
    
    # Create silence tensor with the same shape as waveform except for the last dimension
    silence_shape = list(waveform.shape)
    silence_shape[-1] = padding_samples  # Replace last dim with padding length
    
    silence = torch.zeros(silence_shape).to(waveform.device)
    
    print(f"🔍 Silence shape: {silence.shape}")
    print(f"🔍 Concatenating {waveform.shape} + {silence.shape}")
    
    # Append the silence to the original waveform
    padded_waveform = torch.cat([waveform, silence], dim=-1)
    
    print(f"✅ Padded waveform shape: {padded_waveform.shape}")
    
    return padded_waveform

def load_and_validate_audio(input_data_base64: str) -> tuple[torch.Tensor, int]:
    """Load and validate input audio from base64 string."""
    device = get_device()
    input_audio = None
    try:
        input_data = base64.b64decode(input_data_base64)
        input_audio = io.BytesIO(input_data)
        song, sr = torchaudio.load(input_audio)
        if song.size(0) == 0 or song.size(1) == 0:
            raise AudioProcessingError("Input audio is empty")
        return song.to(device), sr
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {str(e)}")
    finally:
        if input_audio is not None:
            input_audio.close()

def save_audio_to_base64(waveform: torch.Tensor, sample_rate: int) -> str:
    """Save audio tensor to base64 string with proper resource cleanup and dtype handling."""
    output_audio = None
    try:
        # Ensure waveform is in float32 format before saving
        if waveform.dtype != torch.float32:
            waveform = waveform.to(torch.float32)
        
        # Move to CPU if needed
        if waveform.device.type != 'cpu':
            waveform = waveform.cpu()
            
        # Ensure values are in [-1, 1] range
        max_val = torch.abs(waveform).max()
        if max_val > 1.0:
            waveform = waveform / max_val
            
        output_audio = io.BytesIO()
        torchaudio.save(
            output_audio,
            src=waveform,
            sample_rate=sample_rate,
            format='wav'
        )
        output_audio.seek(0)
        return base64.b64encode(output_audio.read()).decode('utf-8')
    finally:
        if output_audio is not None:
            output_audio.close()

def resample_for_model(audio: torch.Tensor, orig_sr: int, model_sr: int = 32000) -> torch.Tensor:
    """Resample audio to model's sample rate if needed using the resampler pool."""
    if orig_sr == model_sr:
        # Ensure consistent dtype even when no resampling is needed
        return audio.to(torch.float32)
    
    device = audio.device
    with tensor_lifecycle() as track_tensor:
        # Convert to float32 before resampling to ensure compatibility
        if audio.dtype != torch.float32:
            audio = audio.to(torch.float32)
            
        resampler = ResamplerPool().get_resampler(orig_sr, model_sr, str(device))
        resampled = track_tensor(resampler(audio))
        return resampled.clone().to(torch.float32)  # Ensure float32 output

def get_model_description(model_name: str, custom_description: Optional[str] = None) -> Optional[str]:
    """Get appropriate description based on model name or custom input."""
    if custom_description:
        return custom_description
    
    model_descriptions = {
        'thepatch/gary_orchestra': "violin, epic, film, piano, strings, orchestra",
        'thepatch/gary_orchestra_2': "violin, epic, film, piano, strings, orchestra"
    }
    return model_descriptions.get(model_name)



# Add this function to ensure consistent dtype throughout the pipeline
def ensure_float32(tensor):
    """Ensure tensor is float32 and on the correct device."""
    if tensor is None:
        return None
    
    if not isinstance(tensor, torch.Tensor):
        return tensor
        
    # Convert to float32 if it's not already
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
        
    return tensor

# Instead of modifying the model, focus on ensuring input/output tensors are float32
def get_model(model_name, device_id=0):
    """Load MusicGen model with kernel caching awareness."""
    cache_key = kernel_cache.get_cache_key(model_name, device_id)
    is_warmed = kernel_cache.has_warmed_kernels(model_name, device_id)
    
    if not is_warmed:
        print(f"[KERNEL] First load of {model_name} - kernels will compile (expect slowdown)")
    else:
        print(f"[KERNEL] Kernels should be cached for {model_name} - expecting faster startup")
    
    model = MusicGen.get_pretrained(model_name)
    
    # If this is first load, do a dummy forward pass to force kernel compilation
    if not is_warmed:
        print(f"[KERNEL] Warming up kernels for {model_name}...")
        device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        
        # Create minimal dummy input
        dummy_audio = torch.randn(1, 1, 16000).to(device=device, dtype=torch.float32)
        
        with torch.no_grad():
            try:
                # Do a minimal generation to compile kernels
                model.set_generation_params(duration=1.0)
                _ = model.generate_continuation(
                    dummy_audio,
                    prompt_sample_rate=32000,
                    descriptions=None,
                    progress=False
                )
                kernel_cache.mark_kernels_warmed(model_name, device_id)
                print(f"[KERNEL] Kernel warmup complete for {model_name}")
            except Exception as e:
                print(f"[KERNEL] Warmup failed (non-fatal): {e}")
        
        # Clean up dummy tensors
        del dummy_audio
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return model

def safe_musicgen_continuation_v2(
    model,
    prompt: torch.Tensor,
    prompt_sample_rate: int,
    descriptions: Optional[List[Optional[str]]] = None,
    progress: bool = True,
    max_retries: int = 2,
    device_id: int = 0
) -> torch.Tensor:
    """
    Simplified safe wrapper for MusicGen's generate_continuation.
    Handles retries for dtype errors without forced synchronization.
    """
    # Validate input type
    if not isinstance(prompt, torch.Tensor):
        raise TypeError(f"Expected prompt to be torch.Tensor, got {type(prompt)}")
    
    print(f"🔍 Input prompt shape: {prompt.shape}")
    
    # Handle different input shapes and ensure mono audio
    if prompt.dim() == 1:
        # [T] -> [1, 1, T]
        prompt = prompt[None, None]
        print(f"🔄 Reshaped from 1D to: {prompt.shape}")
    elif prompt.dim() == 2:
        if prompt.size(0) == 1:
            # [1, T] -> [1, 1, T] (already mono)
            prompt = prompt[None]
            print(f"🔄 Added batch dim: {prompt.shape}")
        elif prompt.size(0) == 2:
            # [2, T] -> [1, 1, T] (convert stereo to mono)
            prompt = ((prompt[0] + prompt[1]) / 2.0)[None, None]
            print(f"🔄 Converted stereo to mono and reshaped: {prompt.shape}")
        else:
            # [C, T] where C > 2 -> take first channel
            prompt = prompt[0:1][None]
            print(f"🔄 Took first channel and added batch dim: {prompt.shape}")
    elif prompt.dim() == 3:
        # [B, C, T] format
        if prompt.size(1) == 1:
            # Already mono, good to go
            print(f"✅ Already proper mono format: {prompt.shape}")
        elif prompt.size(1) == 2:
            # Convert stereo to mono
            prompt = ((prompt[:, 0:1] + prompt[:, 1:2]) / 2.0)
            print(f"🔄 Converted stereo to mono: {prompt.shape}")
        else:
            # Take first channel
            prompt = prompt[:, 0:1]
            print(f"🔄 Took first channel: {prompt.shape}")
    else:
        raise ValueError(f"Unsupported prompt dimensions: {prompt.shape}")
    
    # Final validation: ensure we have [B, 1, T] format
    if prompt.dim() != 3 or prompt.size(1) != 1:
        raise ValueError(f"After reshaping, expected [B, 1, T] format, got {prompt.shape}")
    
    print(f"✅ Final prompt shape for MusicGen: {prompt.shape}")
        
    # Force float32 and pin to device
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    prompt = prompt.to(device=device, dtype=torch.float32)
    
    # Try with retries for dtype errors
    attempt = 0
    last_error = None
    
    while attempt <= max_retries:
        try:
            # Lightweight cleanup between retries if needed
            if attempt > 0:
                print(f"Retry attempt {attempt}/{max_retries}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # Just clear cache, no sync
                gc.collect()
                
                # Try to force float32 on model components (for dtype errors)
                try:
                    if hasattr(model, 'compression_model') and hasattr(model.compression_model, 'to'):
                        model.compression_model = model.compression_model.to(torch.float32)
                    if hasattr(model, 'lm') and hasattr(model.lm, 'to'):
                        model.lm = model.lm.to(torch.float32)
                except Exception as e:
                    print(f"Warning: Could not convert model components: {e}")
            
            # Call generation - let PyTorch handle CUDA async
            output = model.generate_continuation(
                prompt,
                prompt_sample_rate=prompt_sample_rate,
                descriptions=descriptions,
                progress=progress
            )
            
            print(f"✅ Generated output shape: {output.shape}")
            
            # Success! Return as float32
            return output.to(dtype=torch.float32)
                
        except RuntimeError as e:
            error_msg = str(e)
            last_error = e
            
            # Handle specific dtype error
            if "Expected scalar type Float but found Half" in error_msg:
                attempt += 1
                if attempt <= max_retries:
                    print(f"Caught dtype error: {error_msg}")
                    print(f"Will retry with model component conversion...")
                    continue
                else:
                    # Exhausted retries
                    break
            else:
                # Not a dtype error, raise immediately
                raise
    
    # If we get here, we've exhausted retries
    if last_error:
        print(f"Failed after {max_retries} retries. Error: {str(last_error)}")
        raise last_error


# Example initialization in your main script
def initialize_audio_system():
    """
    Initialize the audio processing system with optimized CUDA settings.
    Call this once at application startup.
    """
    # Configure CUDA for stability
    configure_cuda_for_stability()
    
    # Optional: Add environment variable for CUDA module loading
    # This can help prevent automatic mixed precision decisions
    os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
    
    # Optional: Set torch multiprocessing method if using multiple workers
    if torch.cuda.is_available():
        try:
            import torch.multiprocessing as mp
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    print("Audio system initialized with enhanced CUDA stability settings")

# 1. Add this helper function to detect the specific dtype error
def is_dtype_mismatch_error(error_msg):
    """
    Checks if an error message contains the specific dtype mismatch error.
    """
    return isinstance(error_msg, str) and "Expected scalar type Float but found Half" in error_msg

def retry_on_dtype_error(max_retries=2):
    """
    Decorator that retries a function if it fails with a dtype mismatch error.
    
    This specifically targets the "Expected scalar type Float but found Half" error
    that occurs during GPU operations under high load.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            last_error = None
            
            while attempts <= max_retries:
                try:
                    # Execute the function
                    return func(*args, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    last_error = e
                    
                    # Only retry for the specific dtype error
                    if is_dtype_mismatch_error(error_msg):
                        attempts += 1
                        # Configure CUDA before retry for better stability
                        configure_cuda_for_stability()
                        print(f"Caught dtype mismatch error, automatically retrying ({attempts}/{max_retries})")
                        
                        # Ensure we have a clean GPU state before retry
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        if attempts <= max_retries:
                            # Short delay before retry
                            import time
                            time.sleep(0.5)
                            continue
                    
                    # For other errors or if we've exhausted retries, re-raise
                    raise last_error
                    
            # We should never get here, but just in case
            raise last_error
            
        return wrapper
    return decorator

# Example usage in _process_audio_impl
# Example updated _process_audio_impl that uses the robust wrapper
# Implementation for your _process_audio_impl function
def _process_audio_impl_v2(
    input_data_base64: str,
    model_name: str,
    config: AudioConfig,
    progress_callback: Optional[Callable] = None,
    description: Optional[str] = None,
    device_id: int = 0
) -> str:
    """
    Simplified implementation - single context, let PyTorch manage CUDA.
    """
    model = None
    try:
        # Load and validate input
        song, sr = load_and_validate_audio(input_data_base64)
        
        # Use enforced_dtype_context for dtype consistency
        with enforced_dtype_context():
            # Setup device
            device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            
            # Process audio with explicit dtype
            song = song.to(device=device, dtype=torch.float32)
            
            # Resample to model's sample rate
            song_resampled = resample_for_model(song, sr, config.target_sr)
            
            # Preprocess the audio
            processed_waveform = preprocess_audio(song_resampled)
            
            # Wrap/pad audio if needed to ensure sufficient length
            wrapped_waveform = wrap_audio_if_needed(
                processed_waveform,
                config.target_sr,
                config.prompt_duration + config.output_duration
            )
            
            # FIX: Take the FIRST X seconds as prompt (not the last!)
            prompt_waveform = wrapped_waveform[..., :int(config.prompt_duration * config.target_sr)]
            
            # Initialize model
            model = MusicGen.get_pretrained(model_name)
            if progress_callback:
                model.set_custom_progress_callback(progress_callback)
            
            model.set_generation_params(
                use_sampling=True,
                top_k=config.top_k,
                top_p=0.0,
                temperature=config.temperature,
                duration=config.output_duration,
                cfg_coef=config.cfg_coef
            )
            
            final_description = get_model_description(model_name, description)
            
            # Generate - safe_musicgen_continuation_v2 handles retries
            output = safe_musicgen_continuation_v2(
                model,
                prompt_waveform,
                prompt_sample_rate=config.target_sr,
                descriptions=[final_description] if final_description else None,
                progress=True,
                device_id=device_id
            )
            
            if output is None or output.size(0) == 0:
                raise AudioProcessingError("Generated output is empty")
            
            # Resample output back to original sample rate if needed
            if sr != config.target_sr:
                output = resample_for_model(output, config.target_sr, sr)
            
            # Return as base64 - output already includes the prompt + continuation
            return save_audio_to_base64(output.squeeze(0).cpu(), sr)
    
    except Exception as e:
        error_msg = f"Audio processing failed: {str(e)}"
        print(error_msg)
        raise AudioProcessingError(error_msg) from e
    
    finally:
        # Cleanup model
        if model is not None:
            del model
        aggressive_cpu_cleanup()  # ← Use this instead of clean_gpu_memory()
        
        # Lightweight cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Similarly update _continue_music_impl to use the wrapper
def _continue_music_impl_v2(
    input_data_base64: str,
    model_name: str,
    config: AudioConfig,
    progress_callback: Optional[Callable] = None,
    description: Optional[str] = None,
    device_id: int = 0
) -> str:
    """
    Simplified music continuation implementation.
    Single context, no forced synchronization, let PyTorch manage CUDA.
    """
    model = None
    try:
        # Load and validate input
        song, sr = load_and_validate_audio(input_data_base64)
        
        # Use enforced_dtype_context for dtype consistency
        with enforced_dtype_context():
            # Setup device
            device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            
            # Force float32 and move to device
            song = song.to(device=device, dtype=torch.float32)
            
            # Ensure stereo
            if song.size(0) == 1:
                song = song.repeat(2, 1).to(dtype=torch.float32)
            
            # Resample to model's sample rate
            song_resampled = resample_for_model(song, sr, config.target_sr).to(dtype=torch.float32)
            
            # Extract prompt from the end of the audio
            prompt_waveform = song_resampled[..., -int(config.prompt_duration * config.target_sr):].to(dtype=torch.float32)
            
            # Initialize model
            model = MusicGen.get_pretrained(model_name)
            if progress_callback:
                model.set_custom_progress_callback(progress_callback)
            
            model.set_generation_params(
                use_sampling=True,
                top_k=config.top_k,
                top_p=0.0,
                temperature=config.temperature,
                duration=config.output_duration,
                cfg_coef=config.cfg_coef
            )
            
            final_description = get_model_description(model_name, description)
            
            print(f"Generating continuation with description: {final_description}")
            
            # Generate continuation - safe_musicgen_continuation_v2 handles retries
            output = safe_musicgen_continuation_v2(
                model,
                prompt_waveform,
                prompt_sample_rate=config.target_sr,
                descriptions=[final_description] if final_description else None,
                progress=True,
                device_id=device_id
            )
            
            # Ensure output is float32 and proper shape
            output = output.to(dtype=torch.float32)
            if output.dim() == 3:
                output = output.squeeze(0)
            
            # Resample output back to original sample rate if needed
            if sr != config.target_sr:
                output = resample_for_model(output, config.target_sr, sr).to(dtype=torch.float32)
            
            # Get the original audio minus the prompt section
            original_minus_prompt = song[..., :-int(config.prompt_duration * sr)].to(dtype=torch.float32)
            
            # Match channels between original and generated
            if original_minus_prompt.size(0) != output.size(0):
                output = output.repeat(original_minus_prompt.size(0), 1).to(dtype=torch.float32)
            
            # Combine original audio (minus prompt) with generated continuation
            combined_waveform = torch.cat([original_minus_prompt, output], dim=1).to(dtype=torch.float32)
            
            # Return as base64
            return save_audio_to_base64(combined_waveform, sr)
    
    except Exception as e:
        error_msg = f"Music continuation failed: {str(e)}"
        print(error_msg)
        raise AudioProcessingError(error_msg) from e
    
    finally:
        # Cleanup model
        if model is not None:
            del model
        
        aggressive_cpu_cleanup()  # ← Use this instead of clean_gpu_memory()
        
        # Lightweight cleanup - no forced synchronization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Backwards compatibility wrappers
def process_audio(
    input_data_base64: str,
    model_name: str,
    progress_callback: Optional[Callable] = None,
    prompt_duration: int = 6,
    top_k: int = 250,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    description: Optional[str] = None,
    device_id: int = 0
) -> str:
    """
    Enhanced backwards-compatible wrapper for audio processing.
    Includes CUDA stability improvements for high concurrent loads.
    """
    # Ensure CUDA is configured for stability (safe to call multiple times)
    if not hasattr(process_audio, '_initialized'):
        configure_cuda_for_stability()
        process_audio._initialized = True
        
    config = AudioConfig(
        prompt_duration=prompt_duration,
        top_k=top_k,
        temperature=temperature,
        cfg_coef=cfg_coef
    )
    return _process_audio_impl_v2(
        input_data_base64,
        model_name,
        config,
        progress_callback,
        description,
        device_id
    )

# Apply the decorator AFTER defining the function
process_audio = retry_on_dtype_error(max_retries=2)(process_audio)

def continue_music(
    input_data_base64: str,
    model_name: str,
    progress_callback: Optional[Callable] = None,
    prompt_duration: int = 6,
    top_k: int = 250,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    description: Optional[str] = None,
    device_id: int = 0
) -> str:
    """
    Enhanced backwards-compatible wrapper for music continuation.
    Includes CUDA stability improvements for high concurrent loads.
    """
    # Ensure CUDA is configured for stability (safe to call multiple times)
    if not hasattr(continue_music, '_initialized'):
        configure_cuda_for_stability()
        continue_music._initialized = True
    
    config = AudioConfig(
        prompt_duration=prompt_duration,
        top_k=top_k,
        temperature=temperature,
        cfg_coef=cfg_coef
    )
    return _continue_music_impl_v2(
        input_data_base64,
        model_name,
        config,
        progress_callback,
        description,
        device_id
    )

# Apply the decorator AFTER defining the function
continue_music = retry_on_dtype_error(max_retries=2)(continue_music)