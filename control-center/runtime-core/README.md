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
on port 8003. for a DAW on another machine, use SSH port forwarding for now;
the services listen on loopback.

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
`~/.local/share/gary4local/headless/` for runtimes and models.

standard installations share desktop's SA3/carey LoRA library, including
training history, checkpoint selections and dice prompts. it follows the
storage location selected in desktop. model downloads and runtime settings
remain separate. stop a service in one interface before using its library
from the other.

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
