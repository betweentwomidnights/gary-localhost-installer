Gary4JUCE Local Backend Installer
=====================================

This installer sets up Gary4JUCE backend services to run locally on your machine.

IMPORTANT: This is for advanced users who want to run AI audio generation 
services locally instead of using the default remote backend.

What this installer does:
1. Installs Gary4JUCE backend services in Program Files
2. Downloads and configures AI models (Gary, Terry, Jerry)
3. Sets up Python environments and dependencies  
4. Provides a Control Center to manage local services
5. Runs services on localhost ports (8000, 8002, 8005)

After installation:
1. Launch "Gary4JUCE Control Center" from Start Menu
2. Click "Start All Services" to run local backend
3. Configure your Gary4JUCE VST3 plugin to use localhost:
   - Gary (MusicGen): localhost:8000
   - Terry (MelodyFlow): localhost:8002  
   - Jerry (Stable Audio): localhost:8005

System Requirements:
- Windows 10/11 (64-bit)
- 16GB+ RAM recommended
- 15GB+ free disk space for AI models
- NVIDIA GPU strongly recommended (CUDA support)
- Python 3.10+ (will be installed if needed)

Default VST3 Usage (No Installation Required):
If you just want to use Gary4JUCE VST3 with remote backend,
you don't need this installer - just use the VST3 zip file.

Local Backend Benefits:
- Works offline
- Faster generation (no network latency)  
- Privacy (audio stays on your machine)
- Customize model settings

For support: https://github.com/betweentwomidnights
