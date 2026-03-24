from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
import time
import io
import tempfile
import os
from audiocraft.models import MelodyFlow
import gc
from variations import VARIATIONS
import psutil
import logging
from contextlib import contextmanager

import redis

# In both g4l_localhost.py AND localhost_melodyflow.py
import tempfile

# Use a consistent shared temp directory
SHARED_TEMP_DIR = os.path.join(tempfile.gettempdir(), "gary4juce_shared")
os.makedirs(SHARED_TEMP_DIR, exist_ok=True)

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# Add progress callback that writes to Redis
def redis_progress_callback(session_id, current, total):
    if session_id:
        progress_percent = int((current / total) * 100)
        try:
            redis_client.setex(f"progress:{session_id}", 3600, str(progress_percent))
        except Exception as e:
            print(f"Redis progress update failed: {e}")

app = Flask(__name__)
CORS(app)

# Global variables
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('melodyflow')

from flask import after_this_request

def unload_model():
    global model
    if model is not None:
        try:
            del model
        except:
            pass
        model = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
        torch.cuda.empty_cache()
        # Optional extra cleanup on some systems:
        try:
            torch.cuda.ipc_collect()
        except:
            pass
    gc.collect()

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

@contextmanager
def resource_cleanup():
    """Context manager to ensure proper cleanup of GPU resources."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

def load_model():
    """Initialize the MelodyFlow model."""
    global model
    if model is None:
        print("Loading MelodyFlow model...")
        model = MelodyFlow.get_pretrained('facebook/melodyflow-t24-30secs', device=DEVICE)
    return model

def load_audio_from_file(file_path: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from file path."""
    try:
        if not os.path.exists(file_path):
            raise AudioProcessingError(f"Audio file not found: {file_path}")
        
        waveform, sr = torchaudio.load(file_path)

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(device)

    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio from file: {str(e)}")

def find_max_duration(model: MelodyFlow, waveform: torch.Tensor, sr: int = 32000, max_token_length: int = 750) -> tuple:
    """Binary search to find maximum duration that produces tokens under the limit."""
    min_seconds = 1
    max_seconds = waveform.shape[-1] / sr
    best_duration = min_seconds
    best_tokens = None

    while max_seconds - min_seconds > 0.1:
        mid_seconds = (min_seconds + max_seconds) / 2
        samples = int(mid_seconds * sr)
        test_waveform = waveform[..., :samples]

        try:
            tokens = model.encode_audio(test_waveform)
            token_length = tokens.shape[-1]

            if token_length <= max_token_length:
                best_duration = mid_seconds
                best_tokens = tokens
                min_seconds = mid_seconds
            else:
                max_seconds = mid_seconds

        except Exception as e:
            max_seconds = mid_seconds

    return best_duration, best_tokens

def process_audio(waveform: torch.Tensor, variation_name: str, 
                 custom_flowstep: float = None, solver: str = "euler", 
                 custom_prompt: str = None, session_id: str = None, 
                 progress_callback = None) -> torch.Tensor:
    """Process audio with selected variation."""
    
    try:
        if variation_name not in VARIATIONS:
            raise AudioProcessingError(f"Unknown variation: {variation_name}")
        
        config = VARIATIONS[variation_name].copy()
        flowstep = custom_flowstep if custom_flowstep is not None else config['default_flowstep']
        
        if custom_prompt is not None:
            config['prompt'] = custom_prompt

        with resource_cleanup():
            # Load model
            current_model = load_model()

            # Find valid duration and get tokens
            max_valid_duration, tokens = find_max_duration(current_model, waveform)
            config['duration'] = max_valid_duration

            # Override steps and regularization based on solver
            if solver.lower() == "midpoint":
                steps = 64
                use_regularize = False
            else:  # Default to euler
                steps = config['steps']
                use_regularize = True

            # Set model parameters
            current_model.set_generation_params(
                solver=solver.lower(),
                steps=steps,
                duration=config['duration'],
            )

            current_model.set_editing_params(
                solver=solver.lower(),
                steps=steps,
                target_flowstep=flowstep,
                regularize=use_regularize,
                regularize_iters=2 if use_regularize else 0,
                keep_last_k_iters=1 if use_regularize else 0,
                lambda_kl=0.2 if use_regularize else 0.0,
            )

            # Set up progress callback if provided
            if progress_callback and session_id:
                def model_progress_callback(elapsed_steps: int, total_steps: int):
                    progress_callback(session_id, elapsed_steps, total_steps)
                current_model._progress_callback = model_progress_callback

            edited_audio = current_model.edit(
                prompt_tokens=tokens,
                descriptions=[config['prompt']],
                src_descriptions=[""],
                progress=True,
                return_tokens=True
            )

            return edited_audio[0][0]

    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")

@app.route('/transform', methods=['POST'])
def transform_audio():
    """Handle audio transformation requests - simplified for localhost."""
    output_file_path = None  # <-- declare early so cleanup hook can see it

    try:
        # Aggressive localhost cleanup: unload model after each request
        AGGRESSIVE_UNLOAD = os.environ.get("MELODYFLOW_UNLOAD_EACH_REQUEST", "1") == "1"

        if AGGRESSIVE_UNLOAD:
            @after_this_request
            def _cleanup(response):
                # Remove generated WAV
                try:
                    if output_file_path and os.path.exists(output_file_path):
                        os.remove(output_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete output wav: {e}")

                # Unload model + clear GPU
                unload_model()
                return response

        session_id = None

        # ---------------------------------------------------------------------
        # Handle both multipart form-data and JSON
        # ---------------------------------------------------------------------
        if request.content_type and 'multipart/form-data' in request.content_type:
            # File upload mode
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No audio file selected'}), 400
            
            variation = request.form.get('transformation_type', request.form.get('variation'))
            custom_flowstep = request.form.get('flowstep')
            solver = request.form.get('solver', 'euler')
            custom_prompt = request.form.get('prompt', request.form.get('custom_prompt'))
            session_id = request.form.get('session_id')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                temp_file_path = tmp_file.name
            
            try:
                input_waveform = load_audio_from_file(temp_file_path)
            finally:
                os.unlink(temp_file_path)

        else:
            # JSON mode
            data = request.get_json()
            if not data or 'variation' not in data:
                return jsonify({'error': 'Missing required data'}), 400
            
            variation = data['variation']
            custom_flowstep = data.get('flowstep')
            solver = data.get('solver', 'euler')
            custom_prompt = data.get('custom_prompt')
            session_id = data.get('session_id')

            if 'audio_file_path' in data and data['audio_file_path']:
                audio_file_path = data['audio_file_path']
                input_waveform = load_audio_from_file(audio_file_path)
            else:
                return jsonify({'error': 'No audio data provided'}), 400

        # ---------------------------------------------------------------------
        # Validate parameters
        # ---------------------------------------------------------------------
        if custom_flowstep is not None:
            try:
                custom_flowstep = float(custom_flowstep)
                if custom_flowstep <= 0:
                    return jsonify({'error': 'Flowstep must be positive'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid flowstep value'}), 400
        
        if solver not in ['euler', 'midpoint']:
            return jsonify({'error': 'Invalid solver. Must be "euler" or "midpoint"'}), 400

        # ---------------------------------------------------------------------
        # Process audio
        # ---------------------------------------------------------------------
        processed_waveform = process_audio(
            input_waveform,
            variation,
            custom_flowstep,
            solver,
            custom_prompt,
            session_id=session_id,
            progress_callback=redis_progress_callback
        )

        import uuid
        output_filename = f"output_{session_id}_{uuid.uuid4().hex[:8]}.wav"
        output_file_path = os.path.join(SHARED_TEMP_DIR, output_filename)

        torchaudio.save(output_file_path, processed_waveform.cpu(), 32000)

        # Explicit tensor cleanup (before model unload)
        del input_waveform
        del processed_waveform

        return send_file(
            output_file_path,
            as_attachment=True,
            download_name='transformed_audio.wav',
            mimetype='audio/wav'
        )

    except AudioProcessingError as e:
        return jsonify({'error': str(e)}), e.status_code

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/variations', methods=['GET'])
def get_variations():
    """Return list of available variations."""
    try:
        variations_list = list(VARIATIONS.keys())
        variations_with_details = {
            name: {
                'prompt': VARIATIONS[name]['prompt'],
                'flowstep': VARIATIONS[name]['flowstep']
            } for name in variations_list
        }
        return jsonify({
            'variations': variations_with_details
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch variations: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        model_loaded = model is not None
        gpu_available = torch.cuda.is_available()
        
        status = {
            'status': 'healthy' if gpu_available else 'degraded',
            'gpu_available': gpu_available,
            'model_loaded': model_loaded,
        }
        
        return jsonify(status), 200 if status['status'] == 'healthy' else 503
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=False)