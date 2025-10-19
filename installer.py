import os
import sys
import subprocess
import requests
import shutil
import time
from pathlib import Path
import psutil
import zipfile
import stat

def on_rm_error(func, path, exc_info):
    # Windows sometimes marks .git objects read-only
    os.chmod(path, stat.S_IWRITE)
    func(path)

class Gary4JUCEInstaller:
    def __init__(self):
        # Program directory (where the installer executable is)
        self.base_dir = Path.cwd()
        
        # IMPORTANT: Move services to user writable directory
        import os
        if os.name == 'nt':  # Windows
            user_data_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
            self.services_dir = user_data_dir / "Gary4JUCE" / "services"
        else:
            # For other platforms, use a standard user directory
            self.services_dir = Path.home() / ".gary4juce" / "services"
        
        # Service subdirectories (now in user space)
        self.gary_dir = self.services_dir / "gary"
        self.melodyflow_dir = self.services_dir / "melodyflow"
        self.stable_audio_dir = self.services_dir / "stable-audio"

        # KEEP EXISTING: Detect if we're running as a packaged executable
        self.is_packaged = getattr(sys, 'frozen', False)
        self.python_executable = self.find_python_executable()
        
        # Service configurations (unchanged)
        self.services = {
            "gary": {
                "repo": "https://github.com/betweentwomidnights/gary-backend-combined",
                "branch": "localhost-installer",
                "torch_version": "2.1.0",
                "port": 8000
            },
            "melodyflow": {
                "repo": "https://github.com/betweentwomidnights/melodyflow", 
                "branch": "localhost-installer",
                "torch_version": "2.4.0",
                "port": 8002
            },
            "stable-audio": {
                "repo": "https://github.com/betweentwomidnights/stable-audio-api",
                "branch": "main",
                "torch_version": "2.5.0",
                "port": 8005
            }
        }
        
        # Debug output to confirm directories
        print(f"DEBUG: Program directory (executables): {self.base_dir}")
        print(f"DEBUG: Services directory (writable): {self.services_dir}")

    def log(self, message, level="INFO"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def find_python_executable(self):
        """Find the system Python executable."""
        if not self.is_packaged:
            # If running as script, use sys.executable as before
            return sys.executable
        
        # When packaged, we need to find system Python
        python_candidates = [
            "python",           # Try python in PATH first
            "python3",          # Linux/Mac style
            "py",               # Python Launcher on Windows
        ]
        
        # Also check common Windows Python locations
        if os.name == 'nt':  # Windows
            # Check for Python from Microsoft Store
            store_python = Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "python.exe"
            if store_python.exists():
                python_candidates.insert(0, str(store_python))
            
            # Check for system-wide Python installation of 3.11.9
            for base_path in [Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python311"]:
                python_exe = base_path / "python.exe"
                if python_exe.exists():
                    python_candidates.insert(0, str(python_exe))
        
        # Test each candidate
        for candidate in python_candidates:
            try:
                result = subprocess.run([candidate, "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0 and "Python 3.11" in result.stdout:
                    # Verify it's Python 3.11.x
                    version_line = result.stdout.strip()
                    version_parts = version_line.split()[1].split('.')
                    major, minor = int(version_parts[0]), int(version_parts[1])
                    
                    if major == 3 and minor == 11:
                        self.log(f"✅ Found Python {version_line.split()[1]} at: {candidate}")
                        return candidate
                    else:
                        self.log(f"⚠️  Python {version_line.split()[1]} found, but needs to be 3.11.x")
                        
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        # If we get here, no suitable Python was found — try auto-install
        self.log("⚠️ No suitable Python 3.11 found. Attempting to auto-install Python 3.11.9...")

        try:
            self.auto_install_python_311()
            time.sleep(2)  # Allow PATH to update/register Python install
            return self.find_python_executable()  # Retry detection after install
        except Exception as e:
            raise Exception(
                f"❌ Auto-install failed: {e}\n"
                "Please install Python 3.11.9 manually from:\n"
                "https://www.python.org/downloads/release/python-3119/\n"
                "or\n"
                "https://www.python.org/ftp/python/3.11.9/\n"
                "Then rerun this installer."
            )
    
    def auto_install_python_311(self):
        import tempfile, urllib.request, hashlib

        self.log("📥 Downloading Python 3.11.9 installer...")

        url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
        installer_path = Path(tempfile.gettempdir()) / "python-3.11.9-amd64.exe"

        # Download with detailed progress
        try:
            self.log(f"Downloading from: {url}")
            self.log(f"Saving to: {installer_path}")
            
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = round((block_num * block_size / total_size) * 100, 1)
                    if block_num % 100 == 0:  # Log every 10MB or so
                        self.log(f"Download progress: {percent}%")
            
            urllib.request.urlretrieve(url, installer_path, reporthook=show_progress)
            
            # Verify download
            if not installer_path.exists():
                raise Exception("Installer file does not exist after download")
            
            file_size = installer_path.stat().st_size
            self.log(f"✅ Downloaded {file_size:,} bytes to {installer_path}")
            
            # Verify it's actually an executable
            with open(installer_path, 'rb') as f:
                header = f.read(2)
                if header != b'MZ':  # DOS/Windows executable header
                    raise Exception("Downloaded file is not a valid Windows executable")
            
        except Exception as e:
            raise Exception(f"Failed to download Python installer: {e}")

        # Try multiple installation methods
        installation_successful = False
        
        # Method 1: Try interactive first to see if it works at all
        self.log("🔍 Testing installer with interactive mode first...")
        try:
            test_result = subprocess.run([str(installer_path), "/?"], 
                                    capture_output=True, text=True, timeout=10)
            self.log(f"Installer help exit code: {test_result.returncode}")
            if test_result.stdout:
                self.log(f"Installer stdout: {test_result.stdout[:200]}...")
            if test_result.stderr:
                self.log(f"Installer stderr: {test_result.stderr[:200]}...")
        except Exception as e:
            self.log(f"⚠️ Installer test failed: {e}")
        
        # Method 2: Try the simplest possible silent install
        self.log("⚙️ Attempting Method 1: Minimal silent installation...")
        try:
            simple_args = [str(installer_path), "/S", "/v/qn"]  # NSIS-style silent install
            
            result = subprocess.run(
                simple_args,
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(installer_path.parent)
            )
            
            self.log(f"Method 1 exit code: {result.returncode}")
            if result.stdout:
                self.log(f"Method 1 stdout: {result.stdout}")
            if result.stderr:
                self.log(f"Method 1 stderr: {result.stderr}")
                
            if result.returncode == 0:
                installation_successful = True
                
        except subprocess.TimeoutExpired:
            self.log("❌ Method 1 timed out")
        except Exception as e:
            self.log(f"❌ Method 1 failed: {e}")
        
        # Method 3: Try Windows Installer database mode
        if not installation_successful:
            self.log("⚙️ Attempting Method 2: Windows Installer mode...")
            try:
                msi_args = [
                    "msiexec", "/i", str(installer_path),
                    "/quiet", "/norestart",
                    f"TARGETDIR={Path.home() / 'AppData' / 'Local' / 'Programs' / 'Python' / 'Python311'}",
                    "ALLUSERS=0"
                ]
                
                result = subprocess.run(
                    msi_args,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                self.log(f"Method 2 exit code: {result.returncode}")
                if result.stdout:
                    self.log(f"Method 2 stdout: {result.stdout}")
                if result.stderr:
                    self.log(f"Method 2 stderr: {result.stderr}")
                    
                if result.returncode == 0:
                    installation_successful = True
                    
            except Exception as e:
                self.log(f"❌ Method 2 failed: {e}")
        
        # Method 4: Original method with maximum logging
        if not installation_successful:
            self.log("⚙️ Attempting Method 3: Original method with detailed logging...")
            
            install_args = [
                str(installer_path),
                "/quiet",
                "/passive", 
                "/norestart",
                "InstallAllUsers=0",
                "PrependPath=1",
                "Include_launcher=1",
                "Include_pip=1",
                "Include_test=0",
                "Include_tcltk=0",
                "Include_dev=0"
            ]
            
            self.log(f"Running command: {' '.join(install_args)}")
            
            try:
                # Run without CREATE_NO_WINDOW to see if there are any visible errors
                result = subprocess.run(
                    install_args,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                self.log(f"Method 3 exit code: {result.returncode}")
                if result.stdout:
                    self.log(f"Method 3 stdout: {result.stdout}")
                if result.stderr:
                    self.log(f"Method 3 stderr: {result.stderr}")
                
                # Even if exit code is 0, verify Python was actually installed
                if result.returncode == 0:
                    self.log("🔍 Verifying Python installation...")
                    time.sleep(3)  # Give installation time to complete
                    
                    # Check common Python locations
                    potential_pythons = [
                        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python311" / "python.exe",
                        Path("C:/Python311/python.exe"),
                        Path.home() / "AppData" / "Local" / "Microsoft" / "WindowsApps" / "python.exe"
                    ]
                    
                    python_found = False
                    for py_path in potential_pythons:
                        if py_path.exists():
                            self.log(f"✅ Found Python at: {py_path}")
                            python_found = True
                            break
                    
                    if python_found:
                        installation_successful = True
                    else:
                        self.log("❌ Python executable not found after 'successful' installation")
                    
            except subprocess.TimeoutExpired:
                self.log("❌ Method 3 timed out")
            except Exception as e:
                self.log(f"❌ Method 3 failed: {e}")
        
        # Clean up installer file
        try:
            installer_path.unlink()
            self.log("🧹 Cleaned up installer file")
        except:
            self.log("⚠️ Could not clean up installer file")
        
        if installation_successful:
            self.log("✅ Python 3.11.9 installation appears successful.")
            self.log("⏳ Waiting for system to update...")
            time.sleep(5)
        else:
            # Give detailed failure information
            self.log("❌ All Python installation methods failed.")
            self.log("🔍 Possible causes:")
            self.log("   - Windows Defender or antivirus blocking installation")
            self.log("   - User Account Control (UAC) preventing installation")
            self.log("   - Insufficient permissions")
            self.log("   - Corrupted installer download")
            self.log("   - System policy preventing software installation")
            
            raise Exception(
                "Automatic Python installation failed. Please install Python 3.11 manually from:\n"
                "https://www.python.org/downloads/release/python-3119/\n"
                "or\n"
                "https://www.python.org/ftp/python/3.11.9/\n"
                "Make sure to check 'Add Python to PATH' during installation.\n"
                "Then rerun this installer."
            )
    
    def check_system_requirements(self):
        """Check if system has required components."""
        self.log("🔍 Checking system requirements...")
        
        # Test our Python executable
        try:
            result = subprocess.run([self.python_executable, "--version"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                version_line = result.stdout.strip()
                self.log(f"✅ {version_line} OK")
            else:
                raise Exception("Python version check failed")
        except Exception as e:
            raise Exception(f"Python check failed: {e}")
        
        # Check for git
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            self.log("✅ Git available")
        except:
            raise Exception("Git not found. Please install Git for Windows.")
        
        # Check for CUDA (optional but recommended)
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                self.log("✅ NVIDIA GPU detected")
            else:
                self.log("⚠️  No NVIDIA GPU detected - will use CPU (slow)")
        except:
            self.log("⚠️  nvidia-smi not found - will use CPU (slow)")
        
        # Check Redis (we'll install if needed)
        if not self.check_redis():
            self.log("⚠️  Redis not detected - will install")

    def check_redis(self):
        """Check if Redis is available."""
        try:
            import redis
            client = redis.StrictRedis(host='localhost', port=6379, db=0)
            client.ping()
            return True
        except:
            return False

    def create_directories(self):
        """Create directory structure."""
        self.log("📁 Creating directories...")
        
        # Create services directory in user space (with parents=True for full path)
        self.services_dir.mkdir(parents=True, exist_ok=True)
        self.gary_dir.mkdir(exist_ok=True)
        self.melodyflow_dir.mkdir(exist_ok=True) 
        self.stable_audio_dir.mkdir(exist_ok=True)
        
        self.log(f"✅ Services directory created at: {self.services_dir}")

    def clone_repositories(self):
        """Clone all repositories and verify expected Flask apps exist."""
        self.log("📥 Cloning repositories...")

        # Expected main Flask script for each service
        expected_files = {
            "gary": "g4l_localhost.py",
            "melodyflow": "localhost_melodyflow.py",
            "stable-audio": "api.py"
        }

        for name, config in self.services.items():
            service_dir = getattr(self, f"{name.replace('-', '_')}_dir")
            expected_file = expected_files[name]

            # If service dir exists but is missing .git or the expected file, remove it
            if service_dir.exists():
                if not (service_dir / ".git").exists() or not (service_dir / expected_file).exists():
                    self.log(f"🗑️  Removing incomplete {name} directory...")
                    shutil.rmtree(service_dir, onerror=on_rm_error)
                else:
                    self.log(f"⏭️  {name} already cloned and contains {expected_file}, skipping")
                    continue

            # Clone fresh
            self.log(f"📥 Cloning {name} from {config['repo']} ({config['branch']})...")
            cmd = [
                "git", "clone",
                "-b", config["branch"],
                config["repo"],
                str(service_dir)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Failed to clone {name}: {result.stderr.strip()}")

            # Verify the expected file exists
            if not (service_dir / expected_file).exists():
                raise Exception(
                    f"{name} cloned but missing expected file: {expected_file}. "
                    f"Check branch '{config['branch']}' contents."
                )

            self.log(f"✅ {name} cloned successfully with {expected_file}")

    def create_virtual_environments(self):
        """Create isolated Python environments for each service."""
        self.log("🐍 Creating virtual environments...")
        
        for name in self.services.keys():
            service_dir = getattr(self, f"{name.replace('-', '_')}_dir")
            venv_dir = service_dir / "env"
            
            if venv_dir.exists():
                venv_python = venv_dir / "Scripts" / "python.exe"
                if not venv_python.exists():
                    self.log(f"⚠️  {name} venv exists but is broken (no python.exe), recreating...")
                    shutil.rmtree(venv_dir)
                else:
                    self.log(f"⏭️  {name} venv exists, skipping")
                    continue
                
            self.log(f"🐍 Creating {name} virtual environment...")
            
            # Use our found Python executable instead of sys.executable
            result = subprocess.run([
                self.python_executable, "-m", "venv", str(venv_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to create {name} venv: {result.stderr}")
            
            self.log(f"✅ {name} venv created")

    def install_service_dependencies(self, service_name):
        """Install dependencies for a specific service."""
        config = self.services[service_name]
        service_dir = getattr(self, f"{service_name.replace('-', '_')}_dir")
        venv_python = service_dir / "env" / "Scripts" / "python.exe"
        
        self.log(f"📦 Installing {service_name} dependencies...")
        
        # Upgrade pip first
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        if service_name == "gary":
            self.install_gary_deps(venv_python)
        elif service_name == "melodyflow":
            self.install_melodyflow_deps(venv_python)
        elif service_name == "stable-audio":
            self.install_stable_audio_deps(venv_python)

    def install_gary_deps(self, venv_python):
        """Install Gary service dependencies with safetensors version fix."""
        self.log("📦 Installing Gary (MusicGen) dependencies...")
        
        # Clear any existing PyTorch installations first
        self.log("🧹 Ensuring clean PyTorch installation...")
        try:
            subprocess.run([
                str(venv_python), "-m", "pip", "cache", "purge"
            ], capture_output=True, check=False)
            
            # Uninstall existing packages
            packages_to_remove = ["torch", "torchaudio", "torchvision", "av", "audiocraft", "numpy", "safetensors", "xformers"]
            for package in packages_to_remove:
                subprocess.run([
                    str(venv_python), "-m", "pip", "uninstall", package, "-y"
                ], capture_output=True, check=False)
        except:
            pass
        
        # Install torch 2.1.0 with CUDA support
        self.log("⚡ Installing PyTorch 2.1.0 with CUDA support...")
        subprocess.run([
            str(venv_python), "-m", "pip", "install",
            "torch==2.1.0", "torchaudio==2.1.0", 
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--force-reinstall", "--no-cache-dir"
        ], check=True)
        
        # CRITICAL: Install compatible safetensors version for PyTorch 2.1.0
        self.log("🔒 Installing safetensors<0.6.0 for PyTorch 2.1.0 compatibility...")
        subprocess.run([
            str(venv_python), "-m", "pip", "install", "safetensors<0.6.0",
            "--force-reinstall", "--no-cache-dir"
        ], check=True)
        
        # Install audiocraft's dependencies manually first
        self.log("📦 Installing audiocraft dependencies...")
        audiocraft_deps = [
            "av==12.3.0",  # Using your pinned version instead of 11.0.0
            "einops",
            "flashy>=0.0.1",
            "hydra-core>=1.1",
            "hydra_colorlog",
            "julius",
            "num2words",
            "numpy==1.24.0",  # Using your pinned version
            "sentencepiece",
            "spacy==3.7.6",
            "huggingface_hub",
            "tqdm",
            "transformers==4.39.3",  # Using your pinned version
            "demucs",
            "librosa",
            "soundfile",
            "gradio",
            "torchmetrics",
            "encodec",
            "protobuf",
            "torchvision==0.16.0",
            "torchtext==0.16.0",
            "pesq",
            "pystoi",
            "torchdiffeq",
            "xformers<0.0.23"
        ]
        subprocess.run([str(venv_python), "-m", "pip", "install"] + audiocraft_deps, check=True)
        
        # Now install audiocraft itself WITHOUT dependencies (we already installed them)
        self.log("🎵 Installing audiocraft (no-deps)...")
        subprocess.run([
            str(venv_python), "-m", "pip", "install", "audiocraft", "--no-deps"
        ], check=True)
        
        # Install other Gary dependencies
        self.log("📦 Installing additional Gary dependencies...")
        gary_deps = [
            "flask", "flask-cors", "redis", "pydantic", "requests", "psutil"
        ]
        subprocess.run([str(venv_python), "-m", "pip", "install"] + gary_deps, check=True)
        
        # Verify PyTorch installation
        self.log("🔍 Verifying PyTorch installation...")
        try:
            result = subprocess.run([
                str(venv_python), "-c", 
                "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('uint64 available:', hasattr(torch, 'uint64'))"
            ], capture_output=True, text=True, check=True)
            
            self.log(f"✅ PyTorch verification: {result.stdout.strip()}")
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ PyTorch verification failed: {e}")
            raise Exception("PyTorch verification failed")
        
        # Final audiocraft test
        self.log("🎵 Verifying audiocraft integration...")
        try:
            result = subprocess.run([
                str(venv_python), "-c", 
                "import torch; import audiocraft; print('SUCCESS: Audiocraft + PyTorch integration working')"
            ], capture_output=True, text=True, check=True)
            
            self.log(f"✅ Audiocraft integration: {result.stdout.strip()}")
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Audiocraft integration failed: {e}")
            if e.stderr:
                self.log(f"Error details: {e.stderr}")
            raise Exception("Audiocraft integration failed")
        
        self.log("✅ Gary dependencies installed and verified")

    def install_melodyflow_deps(self, venv_python):
        """Install MelodyFlow dependencies."""
        self.log("📦 Installing MelodyFlow dependencies...")
        
        # Install torch 2.4.0 with compatible versions
        torch_deps = [
            "torch==2.4.0", 
            "torchaudio==2.4.0"
        ]
        
        subprocess.run([
            str(venv_python), "-m", "pip", "install"
        ] + torch_deps + [
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        
        # Install ML dependencies (removed torchvision, torchtext since they're above)
        ml_deps = [
            "av", "einops", "flashy>=0.0.1", "hydra-core>=1.1", "hydra_colorlog",
            "julius", "num2words", "numpy==1.25.2", "sentencepiece", "spacy>=3.6.1",
            "huggingface_hub", "tqdm", "transformers>=4.31.0",
            "demucs", "librosa", "soundfile", "torchmetrics==1.8.0", "encodec", 
            "protobuf", "pesq", "pystoi", "xformers==0.0.27.post2"
        ]
        subprocess.run([str(venv_python), "-m", "pip", "install"] + ml_deps, check=True)
        
        # Install web dependencies  
        web_deps = ["flask", "flask-cors", "redis", "psutil"]
        subprocess.run([str(venv_python), "-m", "pip", "install"] + web_deps, check=True)
        
        self.log("✅ MelodyFlow dependencies installed")

    def install_stable_audio_deps(self, venv_python):
        """Install Stable Audio dependencies.""" 
        self.log("📦 Installing Stable Audio dependencies...")
        
        # Install torch 2.5.0+
        subprocess.run([
            str(venv_python), "-m", "pip", "install", 
            "torch>=2.5.0", "torchaudio>=2.5.0",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        
        # Install ML dependencies
        ml_deps = [
            "einops", "transformers", "accelerate", "diffusers", 
            "scipy", "librosa", "soundfile", "huggingface-hub"
        ]
        subprocess.run([str(venv_python), "-m", "pip", "install"] + ml_deps, check=True)
        
        # Install web dependencies
        subprocess.run([str(venv_python), "-m", "pip", "install", "flask", "flask-cors"], check=True)
        
        # Clone and install stable-audio-tools
        self.log("📦 Installing stable-audio-tools...")
        subprocess.run([
            str(venv_python), "-m", "pip", "install",
            "git+https://github.com/Stability-AI/stable-audio-tools.git"
        ], check=True)
        
        self.log("✅ Stable Audio dependencies installed")

    def setup_redis(self):
        """Setup Redis if not available."""
        if self.check_redis():
            self.log("✅ Redis already available")
            return
            
        self.log("🔧 Setting up Redis...")
        # For Windows, we'll provide instructions rather than installing
        self.log("⚠️  Please install Redis for Windows:")
        self.log("   1. Download from: https://github.com/MicrosoftArchive/redis/releases")
        self.log("   2. Or use Docker: docker run -d -p 6379:6379 redis")
        self.log("   3. Or use Windows Subsystem for Linux (WSL)")

    def download_redis_windows(self):
        """Download and setup Redis for Windows."""
        self.log("📥 Setting up Redis for Windows...")
        
        redis_dir = self.base_dir / "redis"
        redis_exe = redis_dir / "redis-server.exe"
        
        if redis_exe.exists():
            self.log("✅ Redis already installed")
            return
        
        redis_dir.mkdir(exist_ok=True)
        
        # Download Redis Windows binaries
        redis_url = "https://github.com/MicrosoftArchive/redis/releases/download/win-3.0.504/Redis-x64-3.0.504.zip"
        redis_zip = redis_dir / "redis.zip"
        
        self.log("📥 Downloading Redis Windows binaries...")
        response = requests.get(redis_url, stream=True)
        with open(redis_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract Redis
        import zipfile
        with zipfile.ZipFile(redis_zip, 'r') as zip_ref:
            zip_ref.extractall(redis_dir)
    
        # Move files from subfolder to root
        extracted_folder = redis_dir / "Redis-x64-3.0.504"  # Likely folder name
        if extracted_folder.exists():
            for item in extracted_folder.iterdir():
                shutil.move(str(item), redis_dir)
            extracted_folder.rmdir()
        
        # Cleanup zip
        redis_zip.unlink()
        
        self.log("✅ Redis installed successfully")

    def start_redis(self):
        """Start Redis server."""
        redis_exe = self.base_dir / "redis" / "redis-server.exe"
        if not redis_exe.exists():
            raise Exception("Redis not found! Run installer first.")
        
        self.log("🔄 Starting Redis server...")
        redis_process = subprocess.Popen([str(redis_exe)], cwd=str(redis_exe.parent))
        time.sleep(3)  # Give Redis time to start
        
        # Test connection
        try:
            import redis
            client = redis.StrictRedis(host='localhost', port=6379, db=0)
            client.ping()
            self.log("✅ Redis server started successfully")
            return redis_process
        except:
            raise Exception("Failed to connect to Redis after startup")

    def setup_hf_token(self):
        """Setup HuggingFace token for Stable Audio."""
        self.log("🔑 Setting up HuggingFace token...")
        
        # FALLBACK TOKEN - Replace with your actual token
        # This provides access to stable-audio-open-small for all users
        FALLBACK_HF_TOKEN = "token_go_here_idk_why_some_machines_dont_pick_it_up_from_env"
        
        hf_token = None
        
        # First check system environment
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            self.log("✅ Found HF_TOKEN in system environment")
        
        # If not in system env, check .env file
        if not hf_token:
            env_file = self.base_dir / ".env"
            if env_file.exists():
                try:
                    with open(env_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('HF_TOKEN='):
                                hf_token = line.split('=', 1)[1].strip()
                                self.log("✅ Found HF_TOKEN in .env file")
                                break
                except Exception as e:
                    self.log(f"⚠️ Error reading .env file: {e}")
        
        # If still no token, use the built-in fallback (no user prompt needed!)
        if not hf_token:
            self.log("✅ Using built-in HuggingFace token for stable-audio access")
            hf_token = FALLBACK_HF_TOKEN
        
        # Always save token to .env file for services to use
        # (This ensures services can find it in AppData location)
        env_file_program = self.base_dir / ".env"  # Program Files location
        env_file_services = self.services_dir / ".env"  # AppData services location
        
        for env_file in [env_file_program, env_file_services]:
            try:
                # Make sure directory exists
                env_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(env_file, "w") as f:
                    f.write(f"HF_TOKEN={hf_token}\n")
                self.log(f"✅ HuggingFace token saved to: {env_file}")
            except Exception as e:
                self.log(f"⚠️ Could not save .env file to {env_file}: {e}")
        
        # Validate token format
        if hf_token and hf_token.startswith('hf_') and len(hf_token) > 10:
            self.log("✅ HuggingFace token appears valid")
        else:
            self.log("⚠️ HuggingFace token may be invalid - check format")
            
        return hf_token

    def install_launcher_dependencies(self):
        """Install dependencies needed by the launcher."""
        self.log("📦 Installing launcher dependencies...")
        
        launcher_deps = [
            "PySide6", 
            "python-dotenv", 
            "psutil", 
            "requests"
        ]
        
        try:
            # Use our found Python executable instead of sys.executable
            subprocess.run([
                self.python_executable, "-m", "pip", "install"
            ] + launcher_deps, check=True)
            
            self.log("✅ Launcher dependencies installed")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to install launcher dependencies: {e}")

    def copy_icon_files(self):
        """Verify icon files are available for the launcher."""
        self.log("🎨 Checking for application icons...")
        
        icon_files = ["icon.ico", "icon.png"]
        found_any = False
        
        for icon_name in icon_files:
            icon_path = self.base_dir / icon_name
            if icon_path.exists():
                self.log(f"✅ Found {icon_name}")
                found_any = True
            else:
                self.log(f"⚠️  {icon_name} not found")
        
        if not found_any:
            self.log("⚠️  No icon files found - launcher will use fallback icon")
        
        return found_any

    def verify_launcher_exists(self):
        """Verify gary4juce-control-center.exe exists alongside installer."""
        launcher_path = self.base_dir / "gary4juce-control-center.exe"
        
        if not launcher_path.exists():
            raise Exception(
                "gary4juce-control-center.exe not found! Please ensure the control center "
                "executable is in the same directory as the installer."
            )
        
        self.log("✅ gary4juce-control-center.exe found")
        return True

    # Update your run_installation method:
    def run_installation(self, clean_install=False):
        """Run the complete installation process with recovery options."""
        try:
            if clean_install:
                self.cleanup_installation()
            
            # Show current status
            status = self.check_installation_status()
            self.log("📊 Installation status:")
            for step, completed in status.items():
                if isinstance(completed, dict):
                    for service, done in completed.items():
                        icon = "✅" if done else "❌"
                        self.log(f"   {icon} {step}: {service}")
                else:
                    icon = "✅" if completed else "❌"
                    self.log(f"   {icon} {step}")
            
            self.log("🚀 Starting Gary4JUCE installation...")
            
            # Verify we have everything we need
            self.verify_launcher_exists()  # NEW: Check launcher.py exists
            self.check_system_requirements()
            
            # Setup infrastructure
            self.create_directories()
            self.copy_icon_files()  # NEW: Check icons
            self.download_redis_windows()
            self.clone_repositories()
            self.setup_hf_token()
            self.create_virtual_environments()
            
            # Install dependencies for each service
            for service_name in self.services.keys():
                try:
                    self.install_service_dependencies(service_name)
                except Exception as e:
                    self.log(f"❌ Failed to install {service_name} dependencies: {e}", "ERROR")
                    self.log("💡 Try running with --clean flag to start fresh", "INFO")
                    raise
            
            # Install launcher dependencies
            self.install_launcher_dependencies()  # NEW: Much smaller function!
            
            self.log("🎉 Installation complete!")
            self.log("📋 Next steps:")
            self.log("   1. Run: gary4juce-control-center.exe")
            self.log("   2. Click 'start all services'")
            self.log("   3. Test with your JUCE plugin!")
            self.log(f"📁 Services installed to: {self.services_dir}")
            
        except KeyboardInterrupt:
            self.log("⚠️  Installation cancelled by user", "WARNING")
            return False
        except Exception as e:
            self.log(f"❌ Installation failed: {e}", "ERROR")
            self.log("", "ERROR")
            self.log("🆘 Recovery options:", "INFO")
            self.log("   1. Run again to resume from where it failed", "INFO") 
            self.log("   2. Run with --clean flag to start completely fresh", "INFO")
            self.log("   3. Check the error above and fix any system issues", "INFO")
            return False
        
        return True
    
    def cleanup_installation(self):
        """Clean up all installation artifacts for a fresh start."""
        self.log("🧹 Cleaning up installation...")
        
        dirs_to_remove = [
            self.services_dir,  # Now points to user AppData location
            self.base_dir / "redis"  # Redis stays in program directory
        ]
        
        # No need to remove launcher.py anymore since we're using .exe
        
        for dir_path in dirs_to_remove:
            if dir_path.exists():
                self.log(f"🗑️  Removing {dir_path}")
                shutil.rmtree(dir_path, onerror=on_rm_error)
        
        self.log("✅ Cleanup complete - ready for fresh install")

    def run_uninstall(self):
        """Run uninstall process - cleanup only, no reinstall."""
        try:
            self.log("🗑️ Starting Gary4JUCE uninstall...")
            
            # Stop any running services first
            self.log("🛑 Stopping any running services...")
            try:
                # Try to gracefully stop services using taskkill
                python_processes = [
                    "python.exe", 
                    "pythonw.exe"
                ]
                
                # Kill any Python processes that might be our services
                for proc_name in python_processes:
                    try:
                        # Kill processes with our service script names in command line
                        service_scripts = ["g4l_localhost.py", "localhost_melodyflow.py", "api.py"]
                        for script in service_scripts:
                            cmd = f'taskkill /F /FI "IMAGENAME eq {proc_name}" /FI "COMMANDLINE eq *{script}*"'
                            subprocess.run(cmd, shell=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                    except:
                        pass
                
                # Kill redis processes
                subprocess.run(["taskkill", "/F", "/IM", "redis-server.exe"], 
                            capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
                
                self.log("✅ Services stopped")
                
            except Exception as e:
                self.log(f"⚠️ Warning stopping services: {e}")
            
            # Clean up all directories
            self.cleanup_installation()
            
            # Also clean up any additional user data
            self.cleanup_user_data()
            
            self.log("✅ Uninstall complete!")
            self.log("🎉 Gary4JUCE has been successfully removed from your system.")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Uninstall failed: {e}", "ERROR")
            return False

    def cleanup_user_data(self):
        """Clean up additional user data that might not be in services directory."""
        self.log("🧹 Cleaning up user data...")
        
        try:
            # Clean up any cache or temp files that services might have created
            import tempfile
            temp_logs = Path(tempfile.gettempdir()) / "gary4juce_logs"
            if temp_logs.exists():
                self.log(f"🗑️ Removing temp logs: {temp_logs}")
                shutil.rmtree(temp_logs, onerror=on_rm_error)
            
            # Clean up HuggingFace cache if it was created by our services
            hf_cache_dir = Path.home() / ".cache" / "huggingface"
            if hf_cache_dir.exists():
                # Only remove our specific models, not all HF cache
                for model_dir in hf_cache_dir.glob("*"):
                    if any(keyword in model_dir.name.lower() for keyword in ["musicgen", "stable-audio", "melodyflow"]):
                        try:
                            self.log(f"🗑️ Removing model cache: {model_dir.name}")
                            shutil.rmtree(model_dir, onerror=on_rm_error)
                        except:
                            pass
            
            self.log("✅ User data cleanup complete")
            
        except Exception as e:
            self.log(f"⚠️ Warning during user data cleanup: {e}")

    def check_installation_status(self):
        """Check what's already installed to allow resuming."""
        status = {
            "directories_created": self.services_dir.exists(),
            "redis_downloaded": (self.base_dir / "redis" / "redis-server.exe").exists(),
            "repos_cloned": {},
            "venvs_created": {},
            "deps_installed": {}
        }
        
        for name in self.services.keys():
            service_dir = getattr(self, f"{name.replace('-', '_')}_dir")
            venv_dir = service_dir / "env"
            venv_python = venv_dir / "Scripts" / "python.exe"
            
            status["repos_cloned"][name] = (service_dir / ".git").exists()
            status["venvs_created"][name] = venv_python.exists()
            
            # Quick dependency check
            if venv_python.exists():
                try:
                    result = subprocess.run([
                        str(venv_python), "-c", "import torch; print('ok')"
                    ], capture_output=True, text=True, timeout=10)
                    status["deps_installed"][name] = result.returncode == 0
                except:
                    status["deps_installed"][name] = False
            else:
                status["deps_installed"][name] = False
        
        return status

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gary4JUCE Localhost Installer")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean up previous installation and start fresh")
    parser.add_argument("--uninstall", action="store_true",
                       help="Uninstall Gary4JUCE (cleanup only, no reinstall)")
    parser.add_argument("--status", action="store_true",
                       help="Check current installation status")
    
    args = parser.parse_args()
    
    installer = Gary4JUCEInstaller()
    
    if args.status:
        status = installer.check_installation_status()
        print("📊 Installation Status:")
        for step, completed in status.items():
            if isinstance(completed, dict):
                for service, done in completed.items():
                    icon = "✅" if done else "❌"
                    print(f"   {icon} {step}: {service}")
            else:
                icon = "✅" if completed else "❌"
                print(f"   {icon} {step}")
    elif args.uninstall:
        # Run uninstall process
        success = installer.run_uninstall()
        if success:
            input("Press Enter to exit...")
        else:
            input("Press Enter to exit...")
    else:
        # Normal install/clean install
        success = installer.run_installation(clean_install=args.clean)
        if success:
            input("Press Enter to exit...")
        else:
            input("Press Enter to exit...")