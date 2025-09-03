# gary4juce local backend installer

![installer + control center](./installer_control_center.gif)

> this repo exists for **full transparency**. i can’t afford code signing right now, so everything needed to build the installer/control center is here. you can audit it, reproduce it, and run the same build i provide on gumroad.

there are 3 open source music models (3 separate python environments) contained in here:

* musicgen (continuations using finetunes) aka gary: https://github.com/facebookresearch/audiocraft

* melodyflow (transformations) aka terry: https://huggingface.co/spaces/facebook/Melodyflow

* stable-audio-open-small (12 seconds of bpm-aware generations for drums/instrument loops) aka jerry: https://huggingface.co/stabilityai/stable-audio-open-small


---

## why this exists

ai music tools should run **locally**. subscriptions throttle creativity and your audio shouldn’t live on someone else’s cloud. this project ships a tiny control center you can run from your system tray to start/stop local backends for gary4juce.

if you don’t have the gpu (most folks don’t have 12gb… yet), i host a backend free of charge so anyone (including mac users) can play with the models immediately.

* download the **vst/au**: [https://thepatch.gumroad.com/l/gary4juce](https://thepatch.gumroad.com/l/gary4juce)
* frontend plugin source: [https://github.com/betweentwomidnights/gary4juce](https://github.com/betweentwomidnights/gary4juce)
* combined docker backend (alt path): [https://github.com/betweentwomidnights/gary-backend-combined](https://github.com/betweentwomidnights/gary-backend-combined)
* website (under reconstruction): [https://thecollabagepatch.com](https://thecollabagepatch.com)

---

## what this installer does

* installs the gary4juce backend services in a user-writable location
* downloads/configures ai models (gary / terry / jerry)
* sets up isolated python environments and dependencies
* provides a **control center** (pyside6 gui) to manage services
* runs services on `localhost` ports so the vst can talk to them

### service ports

* **gary** (musicgen): `localhost:8000`
* **terry** (melodyflow): `localhost:8002`
* **jerry** (stable audio): `localhost:8005`

---

## system requirements

* windows 10/11 (64‑bit)
* 16 gb ram recommended
* 15+ gb free disk space (models)
* nvidia gpu w/ \~12 gb vram (cuda) strongly recommended
* python 3.10–3.11

> don’t want to install anything? use the **remote backend** via the vst/au from gumroad.

---

## quick start (remote backend: zero setup)

1. grab the vst/au: [https://thepatch.gumroad.com/l/gary4juce](https://thepatch.gumroad.com/l/gary4juce)
2. load **gary4juce** in your daw (ableton, logic, cubase, reaper, etc.)
3. leave the backend urls default (remote). start playing.

---

## build the installer yourself (local backend)

> this produces an unsigned installer and a single exe called **“gary4juce control center”**.

### prerequisites

* git for windows
* python 3.10 or 3.11
* **inno setup 6** (free): [https://jrsoftware.org/isinfo.php](https://jrsoftware.org/isinfo.php)

### clone + install deps

```powershell
# from an empty folder you like
git clone https://github.com/betweentwomidnights/gary-localhost-installer.git
cd gary-localhost-installer

# optional but recommended: a local venv
python -m venv .venv
./venv/Scripts/Activate

# install runtime + build deps
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### environment config

copy `.env.example` to `.env` and add your hf token. make sure you have gone to huggingface and been allowed access to stable-audio-open-small. it's free. (https://huggingface.co/stabilityai/stable-audio-open-small) the control center loads `.env` on start.

### build

```powershell
python build.py
# choose a mode when prompted:
#   e = executables only
#   f = full build (executables + inno setup installer)
```

* pyinstaller specs are in the repo: `gary4juce-installer.spec`, `gary4juce-control-center.spec`
* build outputs land in `dist/`
* the final installer (if you chose full build) lands in `output/`

---

## run the local backend

1. launch **gary4juce control center** from the start menu (or run the exe directly)
2. click **start all services**
3. in your daw, set the plugin to `local`:

   * gary → `http://localhost:8000`
   * terry → `http://localhost:8002`
   * jerry → `http://localhost:8005`

---

## security + reproducibility (unsigned binaries)

these binaries are **not code‑signed**. to verify what you’re running:

1. read the spec/scripts in this repo
2. build locally using the steps above
3. compare checksums

```powershell
certutil -hashfile .\output\YourInstaller.exe SHA256
```

if your hash matches the published one in a release, you’ve reproduced the build.

---

## troubleshooting

* **missing inno setup**: install from [https://jrsoftware.org/isinfo.php](https://jrsoftware.org/isinfo.php) and re-run `python build.py`.
* **gpu not detected**: ensure nvidia drivers/cuda are installed.
* **line ending noise from git**: this repo ships a `.gitattributes` that normalizes endings. if needed, run `git add --renormalize .` once.

---

## contributing

issues and prs welcome. 

---

## license

mit. see `LICENSE.txt`.
