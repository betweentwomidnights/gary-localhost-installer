# stable-audio-api

this fork is an experimental api that exposes several endpoints for playing with the stable-audio-open-small model.

this goes inside [gary's](https://github.com/betweentwomidnights/gary4live) backend.

the stable-audio-open-small model generates 12 seconds of high-quality audio in under a second, with excellent bpm awareness that opens up lots of interesting use cases for music production workflows.

## features

- **text-to-audio generation** - basic prompt-based audio generation
- **bpm-aware loop generation** - automatically calculates optimal loop lengths based on bpm
- **style transfer** - audio-to-audio style transfer with configurable strength
- **personal riff library** - use your own audio files as style transfer sources
- **intelligent negative prompting** - automatically enhances prompts for drums vs instruments

## requirements

you need access to the stable-audio-open-small model on huggingface:
https://huggingface.co/stabilityai/stable-audio-open-small

create a huggingface account, request access to the model, and get your access token.

## installation

```bash
git clone https://github.com/betweentwomidnights/stable-audio-api
cd stable-audio-api

# set your huggingface token
export HF_TOKEN=your_hf_token_here

# build the docker image
docker build -t thecollabagepatch/stable-gary:latest -f Dockerfile .

# run with docker-compose (recommended)
docker compose up -d
```

the api will be available at `http://localhost:8005`

## model information

you can get detailed model and endpoint information:

```bash
curl http://localhost:8005/model/info
```

example response:
```json
{
  "config": {
    "diffusion_objective": "rf_denoiser",
    "io_channels": 64,
    "max_duration_seconds": 12,
    "sample_rate": 44100,
    "sample_size": 524288
  },
  "device": "cuda",
  "model_name": "stabilityai/stable-audio-open-small",
  "supported_endpoints": {
    "/generate": "Text-to-audio generation",
    "/generate/loop": "BPM-aware loop generation (text or style transfer)",
    "/generate/style-transfer": "Audio-to-audio style transfer"
  },
  "supported_parameters": {
    "bars": {"default": "auto", "options": [1,2,4,8], "type": "int"},
    "cfg_scale": {"default": 6.0, "range": "0-20", "type": "float"},
    "loop_type": {"default": "auto", "options": ["drums","instruments","auto"], "type": "string"},
    "negative_prompt": {"required": false, "type": "string"},
    "prompt": {"required": true, "type": "string"},
    "return_format": {"default": "file", "options": ["file","base64"], "type": "string"},
    "seed": {"default": -1, "note": "-1 for random", "type": "int"},
    "steps": {"default": 8, "range": "1-100", "type": "int"},
    "style_strength": {"default": 0.8, "note": "For style transfer", "range": "0.1-1.0", "type": "float"}
  }
}
```

## endpoints

### health check

```bash
curl http://localhost:8005/health
```

### basic generation

generate audio from text prompt:

```bash
curl -X POST http://localhost:8005/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "lo-fi hip-hop beat with pianos 90bpm",
    "steps": 8,
    "cfg_scale": 1.0,
    "seed": 12345
  }' \
  --output generated_audio.wav
```

### bpm-aware loop generation

the model excels at bpm awareness. loops are automatically sized to fit optimal bar counts based on the specified bpm:

```bash
# generate drum loop
curl -X POST http://localhost:8005/generate/loop \
  -F "prompt=aggressive techno drums 140bpm" \
  -F "loop_type=drums" \
  -F "bars=4" \
  -F "cfg_scale=1.0" \
  --output drums_140bpm.wav

# generate instrument loop  
curl -X POST http://localhost:8005/generate/loop \
  -F "prompt=deep techno bass 140bpm" \
  -F "loop_type=instruments" \
  -F "bars=4" \
  -F "cfg_scale=1.0" \
  --output bass_140bpm.wav
```

when `bars` is not specified, the api automatically selects optimal bar counts:
- 74 bpm: 2 bars (~6.5s)
- 128 bpm: 4 bars (~7.5s)  
- 170 bpm: 8 bars (~11.3s)

### style transfer

transform existing audio using text prompts. works well with style_strength around 0.6-0.8:

```bash
curl -X POST http://localhost:8005/generate/style-transfer \
  -F "audio_file=@input_audio.wav" \
  -F "prompt=aggressive techno 140bpm" \
  -F "style_strength=0.6" \
  -F "steps=8" \
  -F "cfg_scale=6.0" \
  --output style_transfer_output.wav
```

### personal riff library

use your own audio files as style transfer sources. place wav files in the `riffs/` folder with this naming format:

```
riffs/
├── a_74bpm_1.wav
├── asharp_89bpm_2.wav
├── gsharp_128bpm_1.wav
└── f_110bpm_3.wav
```

the riff_manager will automatically select and time-stretch riffs to match your target bpm:

```bash
# see available riffs
curl http://localhost:8005/riffs/available

# generate using a riff
curl -X POST http://localhost:8005/generate/loop-with-riff \
  -F "prompt=dark techno 140bpm" \
  -F "key=gsharp" \
  -F "loop_type=instruments" \
  -F "style_strength=0.8" \
  --output riff_transfer_output.wav
```

## usage in production

this api is designed to work alongside other backends. in our gary-backend-combined setup, we:

- append ableton's global bpm to prompts for grid-aligned generations in gary4live
- use global bpm in jerry-4-loops so drum and instrument loops stack perfectly
- leverage the speed for real-time music production workflows

## claude will get it

just provide the api.py file and claude will understand exactly how to curl these endpoints.

## use cases

- **gary4live**: max for live device focused on manipulating audio directly from ableton.
- **gary4beatbox**: ios/android app. SAOS is used to generate drums for gary (musicgen) to continue.
- **jerry-4-loops**: ios app for generating stackable drum and instrument loops (in active development)  
- **real-time music production**: sub-second generation enables live performance use
- **style transfer experiments**: transform existing audio with prompt guidance

## development

the api includes comprehensive error handling, gpu memory management, and resource cleanup. logs provide detailed generation information including timing and memory usage.

to modify endpoints or add features, edit `api.py` and rebuild the docker image.

## credits

shoutout to zach, zach, zach and everyone else involved in the creation of a model with such incredible generation speed.

based on stable-audio-tools by stability ai: https://github.com/Stability-AI/stable-audio-tools