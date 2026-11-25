# headless gary4local

run gary4local's seven GPU services from a Linux terminal, with model downloads
and LoRA training built in. use gary4juce or the service APIs for generation.
you'll need a GPU with a working Vulkan driver.

## install

download a headless archive from the [latest release](https://github.com/betweentwomidnights/gary-localhost-installer/releases/latest):

- `linux-x64` for an x64 PC.
- `linux-arm64` for a DGX Spark or another arm64 machine.

from the folder containing your download, extract it somewhere permanent:

```sh
mkdir -p ~/apps/gary4local/headless
tar -xzf gary4local-server-linux-x64-experimental.tar.gz -C ~/apps/gary4local/headless
sh ~/apps/gary4local/headless/install-gary.sh
export PATH="$HOME/.local/bin:$PATH"
gary
```

for arm64, use the `linux-arm64` filename in the extraction command.

the installer creates `~/.local/bin/gary`, pointing to the extracted binary.
keep that folder. if `~/.local/bin` isn't already on your PATH, add the export
line to your shell's startup file once.

## start your first service

inside Gary, type:

```text
install sa3
models sa3
start sa3
```

`install` downloads the runtime and checks the GPU. `models` lets you choose
and download a model and precision. Gary manages the storage folders for you.
`start` launches the service and displays its address in the service table.

commands work with or without a leading slash. when a command needs a service,
you can name it or use the arrow-key picker. Enter selects; Esc goes back.

| service name | model |
| --- | --- |
| `sa3` | Stable Audio 3 |
| `gary` | MusicGen |
| `melodyflow` | MelodyFlow |
| `stable-audio` | Stable Audio Open |
| `carey` | ACE-Step |
| `foundation` | Foundation-1 |
| `yuey` | YuE2 |

## everyday commands

| command | what it does |
| --- | --- |
| `start sa3` / `stop sa3` | start or stop one service |
| `start all` / `stop all` | start prepared services or stop running services |
| `status` | show which services are live and their addresses |
| `models sa3` | download, select or remove model variants |
| `logs sa3` | show the service's log-file path |
| `doctor sa3` | check setup when a service won't start |
| `help` | show the available commands |
| `quit` | leave Gary and stop services started by this console |

`start all` skips services that still need runtimes or models. keep Gary open
while using the services it started.

the client ports match desktop gary4local, including SA3 on port 8006 and carey
on port 8003. the services listen on loopback, so only this machine can reach
them. for a DAW on another computer, see
[using Gary from another computer](#using-gary-from-another-computer).

## using Gary from another computer

a lot of people running headless Gary will have their DAW on a different
machine. the services only listen on loopback, so something has to bridge the
two. right now the easiest bridge is [Tailscale](https://tailscale.com).

put both machines on the same tailnet. then, on the machine running Gary,
forward the port of each service you've started:

```sh
for p in 8003 8006 8007 8015; do
  sudo tailscale serve --bg --tcp $p tcp://127.0.0.1:$p
done
```

that example covers carey, SA3, yuey and foundation. swap in the ports for
whatever you're running:

| service | port |
| --- | --- |
| `gary` | 8000 |
| `melodyflow` | 8002 |
| `carey` | 8003 |
| `stable-audio` | 8005 |
| `sa3` | 8006 |
| `yuey` | 8007 |
| `foundation` | 8015 |

`tailscale serve status` should list each forward as `(tailnet only)`.
nothing gets exposed to the public internet; the ports are only reachable
from your tailnet.

then in gary4juce, open gary settings, choose `gary4local host...`, enter the
Gary machine's tailnet name or IP, and switch the backend to local. services
you haven't started just show as offline. the host setting needs a gary4juce
build newer than v5.0.6.

i've tested this with headless Gary on a DGX Spark and gary4juce on a Windows
PC. health checks came back in about 15-30 ms over the tailnet.

`--bg` keeps the forwards running, across reboots too, until you remove them:

```sh
sudo tailscale serve reset
```

`reset` clears every serve config on the machine, not just Gary's. if you use
`tailscale serve` for anything else, check `tailscale serve status` first.

### without Tailscale

SSH forwarding still works. forward each port to the same port on the DAW
machine and leave gary4juce's host empty, so it keeps using this computer:

```sh
ssh -N -L 8006:127.0.0.1:8006 -L 8003:127.0.0.1:8003 you@gary-machine
```

add one `-L` per service. stop any gary4local services on the DAW machine that
use those ports first.

### what's next

we're going to explore a listener built into gary4local next: an opt-in setting
that opens the client ports to your local network, with a pairing token so a
random device on the same wifi can't use your GPU. until that exists, Tailscale
or SSH keeps the services off the open network.

## choosing models

`models` shows the downloaded variants, active precision and download size.
where the runtime supplies a GPU recommendation, Gary uses it to suggest a
precision. stop the service before changing its selection or removing weights.

carey keeps a separate selection for each model, so you can use BF16 for 2B and
Q8_0 for XL, for example. to delete a variant, select it in `models` and choose
**Remove weights**. Gary asks for confirmation.

## LoRAs and training

```text
train lora sa3
train lora carey
```

stop that service first. the wizard asks for your dataset folder, offers
auto-labeling, lets you review or edit sidecars, then walks through training
settings. during sidecar review, `label` fills missing labels and `label all`
relabels every track, with confirmation before replacing existing labels.

use `training` for progress and log paths, and `cancel` to stop at a checkpoint
boundary. completed runs add their LoRA and dice prompts automatically.

to import an existing adapter, use `add lora sa3` or `add lora carey`. SA3 takes
GGUF or safetensors; carey takes a PEFT folder and asks whether it's for 2B or XL.
use `loras sa3` or `loras carey` to browse adapters and select a saved training
checkpoint. stop the service before changing checkpoints, then start it again
to make the adapter available to clients.

## storage and desktop

Gary manages its configuration and model storage automatically. the normal
Linux locations are `~/.config/gary4local/headless/` for configuration and
`~/.local/share/gary4local/headless/` for runtimes.

standard installations share desktop's models and its SA3/carey LoRA library,
including training history, checkpoint selections and dice prompts. both follow
the storage location selected in desktop, so a machine keeps one copy of each
model. on a machine without desktop, Gary uses desktop's default location, and
installing desktop later finds the models already there. runtimes and runtime
settings stay separate, because each app pins its own versions.

a download that the other app is already making waits for it instead of
starting over. removing a model removes it for both apps; a service that's
already running keeps working until it restarts. stop a service in one
interface before using its LoRA library from the other.

installations from before models were shared merge their own models into
desktop's folder the first time Gary starts, or the first time `serve`,
`download-models` or `install` runs. copies desktop already has are deleted,
the rest are moved, and Gary says how much it moved and freed. a file that
differs from desktop's and from the published one is set aside under
`models-unmerged` in the runtime folder instead of being deleted. a profile
with its own `models_dir` keeps using it.

## updating

finish training, stop services and quit Gary. extract the new archive into a
new permanent folder. the installer refuses to overwrite an existing `gary`
command, so inspect `command -v gary` and back up the old generated launcher
before removing it and running the new `install-gary.sh`.

remove only the launcher, not your model or library storage. keep the old
extracted folder until you've checked the update, so its launcher can still
be restored. there isn't an automatic CLI updater yet.

## automation

`gary COMMAND --json` provides the noninteractive interface. the separate
[agent guide](../../docs/headless-agent-workflow.md) covers discovery, explicit
configuration and service supervision. training currently uses the interactive
console. that guide is also bundled as `AGENT-WORKFLOW.md` in the archive.
