import subprocess
import sys
import shutil
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not already installed."""
    try:
        subprocess.run(["pyinstaller", "--version"], 
                      capture_output=True, check=True)
        print("✅ PyInstaller already installed")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("📦 Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], 
                      check=True)
        print("✅ PyInstaller installed")

def verify_files():
    """Verify required files exist before building."""
    required_files = [
        "installer.py",
        "launcher.py", 
        "icon.ico"
    ]
    
    missing_files = []
    for filename in required_files:
        if not Path(filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def create_env_template():
    """Create a template .env file for distribution if it doesn't exist."""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("⚠️  No .env file found - creating template...")
        with open(env_file, "w") as f:
            f.write("# HuggingFace Token for Stable Audio Open Small\n")
            f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
            f.write("HF_TOKEN=your_token_here\n")
        print("✅ Template .env created - UPDATE IT with your actual HF token before building!")
        return False
    else:
        # Check if it has a valid token
        with open(env_file, "r") as f:
            content = f.read()
            if "your_token_here" in content or "HF_TOKEN=" not in content:
                print("⚠️  .env file exists but may not have a valid token")
                print("💡 Make sure HF_TOKEN=hf_... is set with your actual token")
                response = input("Continue anyway? (y/n): ").lower().strip()
                if response != 'y':
                    return False
        
        print("✅ .env file found and will be included in build")
        return True

def create_support_files():
    """Create README and LICENSE files for the installer."""
    
    readme_content = """Gary4JUCE Local Backend Installer
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
"""

    license_content = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

    # Write files
    with open("README.txt", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    with open("LICENSE.txt", "w", encoding="utf-8") as f:
        f.write(license_content)
    
    print("✅ Support files created")

def build_executables():
    """Build both installer and launcher executables."""
    
    if not verify_files():
        return False
    
    # Check/create .env file
    if not create_env_template():
        return False
    
    install_pyinstaller()
    create_support_files()
    
    print("🔨 Building executables...")
    
    # Clean previous builds
    for dir_name in ["build", "dist"]:
        if Path(dir_name).exists():
            shutil.rmtree(dir_name)
    
    # Build installer executable
    print("📦 Building service installer...")
    installer_cmd = [
        "pyinstaller",
        "--onefile",
        "--console", 
        "--icon=icon.ico",
        "--name=gary4juce-installer",
        "--add-data=launcher.py;.",
        "--add-data=icon.ico;.",
        "--add-data=.env;.",  # ← NEW: Include .env file!
        "installer.py"
    ]
    
    if Path("icon.png").exists():
        installer_cmd.insert(-1, "--add-data=icon.png;.")
    
    try:
        subprocess.run(installer_cmd, check=True)
        print("✅ Service installer built (with .env included)")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installer build failed: {e}")
        return False
    
    # Build launcher executable  
    print("🎮 Building control center...")
    launcher_cmd = [
        "pyinstaller",
        "--onefile",
        "--windowed",  # No console for launcher
        "--icon=icon.ico", 
        "--name=gary4juce-control-center",
        "launcher.py"
    ]
    
    try:
        subprocess.run(launcher_cmd, check=True)
        print("✅ Control center built")
    except subprocess.CalledProcessError as e:
        print(f"❌ Launcher build failed: {e}")
        return False
    
    # Show results
    dist_path = Path("dist")
    if dist_path.exists():
        exe_files = list(dist_path.glob("*.exe"))
        print(f"\n🎉 Built {len(exe_files)} executables:")
        for exe in exe_files:
            size_mb = exe.stat().st_size / (1024 * 1024)
            print(f"   • {exe.name} ({size_mb:.1f} MB)")
    
    return True

def build_installer_package():
    """Build the final installer package using Inno Setup."""
    
    # Check if Inno Setup is available
    inno_paths = [
        r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        r"C:\Program Files\Inno Setup 6\ISCC.exe",
        r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe", 
        r"C:\Program Files\Inno Setup 5\ISCC.exe"
    ]
    
    inno_compiler = None
    for path in inno_paths:
        if Path(path).exists():
            inno_compiler = path
            break
    
    if not inno_compiler:
        print("❌ Inno Setup not found!")
        print("📥 Please install Inno Setup from: https://jrsoftware.org/isinfo.php")
        print("   Then re-run this script to create the installer package")
        return False
    
    # Check if the Inno Setup script exists
    iss_file = Path("gary4juce-setup.iss")
    if not iss_file.exists():
        print("❌ gary4juce-setup.iss not found!")
        print("💡 Create the Inno Setup script file first")
        return False
    
    print("📦 Building installer package with Inno Setup...")
    
    try:
        result = subprocess.run([inno_compiler, str(iss_file)], 
                               capture_output=True, text=True, check=True)
        print("✅ Installer package created!")
        
        # Look for the output installer
        output_dir = Path("output")
        if output_dir.exists():
            installer_files = list(output_dir.glob("*.exe"))
            if installer_files:
                installer_file = installer_files[0]
                size_mb = installer_file.stat().st_size / (1024 * 1024)
                print(f"📁 Final installer: {installer_file.name} ({size_mb:.1f} MB)")
                print(f"📍 Location: {installer_file.absolute()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Inno Setup build failed: {e}")
        print("Output:", e.stdout)
        print("Error:", e.stderr)
        return False

def clean_build():
    """Clean up build artifacts."""
    
    dirs_to_clean = ["build", "dist", "__pycache__", "output"]
    files_to_clean = ["*.spec", "README.txt", "LICENSE.txt"]
    
    for dirname in dirs_to_clean:
        if Path(dirname).exists():
            shutil.rmtree(dirname)
            print(f"🧹 Removed {dirname}/")
    
    # Clean spec files and support files
    for pattern in files_to_clean:
        for file_path in Path(".").glob(pattern):
            file_path.unlink()
            print(f"🧹 Removed {file_path}")
    
    print("✅ Build artifacts cleaned")

if __name__ == "__main__":
    print("Gary4JUCE Local Backend Build Tool")
    print("==================================")
    print("This builds the LOCAL BACKEND installer for advanced users.")
    print("Regular users just need the VST3 zip file (remote backend).")
    print("")
    
    choice = input("Build (e)xecutables, (i)nstaller package, (f)ull build, or (c)lean? ").lower().strip()
    
    if choice == 'e':
        success = build_executables()
        if success:
            print("\n🎉 Executables built successfully!")
            print("📋 Next step: Run with 'i' to create installer package")
    
    elif choice == 'i':
        if not Path("dist").exists() or not list(Path("dist").glob("*.exe")):
            print("❌ No executables found. Run with 'e' first to build executables.")
            sys.exit(1)
        
        success = build_installer_package()
        if success:
            print("\n🎉 Professional installer created!")
            print("📋 Ready for distribution!")
    
    elif choice == 'f':
        print("🚀 Full build: executables + installer package")
        
        success = build_executables()
        if not success:
            print("❌ Executable build failed")
            sys.exit(1)
        
        success = build_installer_package() 
        if success:
            print("\n🎉 Complete professional installer ready!")
            print("📋 Your LOCAL BACKEND installer is ready for distribution!")
            print("💡 Users who want local backend will use this installer")
            print("💡 Users who want remote backend just need the VST3 zip")
        else:
            print("⚠️  Executables built but installer package failed")
    
    elif choice == 'c':
        clean_build()
    
    else:
        print("❌ Invalid choice. Use 'e', 'i', 'f', or 'c'")
        sys.exit(1)