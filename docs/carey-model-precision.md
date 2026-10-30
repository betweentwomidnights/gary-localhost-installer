# Carey: choosing which precision renders

Status: design, not built. Branch target: `feature/carey-native-migration` (PR #11).

## The problem

gary4local downloads Carey's models one precision at a time. The models panel already offers every precision of every model (BF16, Q8_0, Q6_K, Q5_K_M, Q4_K_M), with the GPU's recommended one marked. ace-server, however, loads **every GGUF in `models/carey`** into one registry, and gary4juce picks a model by name from `GET /props`. When two precisions of the same model are on disk, gary4juce's `resolveDit` takes the highest.

So downloading XL base BF16 "to try it" next to Q4_K_M makes an 8 GB card render XL in BF16 from then on, with nothing in either app saying so or offering a way back short of deleting the file.

The only "use" control today is on the 2B base row. It sets `Selection.encoding`, the base precision Carey needs in order to start. It doesn't decide what renders.

## What the user gets

In the Carey models panel, every model row (2B and XL base, sft and turbo, and the language models) shows:

- the precision it **renders with**, badged `active`;
- a **use** button on any other precision that is on disk;
- a plain warning when the active precision is unlikely to fit the selected GPU: "may run out of GPU memory on this card". It still lets the user choose it, at their own risk.

Choosing takes effect the next time Carey starts. If Carey is running, gary4local restarts it, as it already stops Carey for downloads, and says so: "Carey restarts to use XL base Q8_0".

gary4juce needs no change. Its `/props` then lists exactly one precision per model, so its base/sft/turbo and 2B/XL buttons reflect the active choices, and `resolveDit`'s highest-precision rule never comes into play locally. acestep.cpp and the remote contract don't change either.

## How it works

### The choice

- **Storage:** `migration::Selection` gains `#[serde(default)] precisions: BTreeMap<String, String>`, from variant (`xl-base`, `turbo`, `lm-1.7B`, …) to tier. Selections saved before this simply have no entries.
- **The 2B base** keeps using `Selection.encoding`, which already means "the base precision Carey needs to start". The base's `use` keeps calling `select_native_model`.
- **Other models** use a new command, `select_native_precision(service_id, variant, tier)`. It requires the component to be on disk, refuses while a native mutation is in progress (`native_mutation_blocker`), stores the choice, and restarts Carey if it's running.

### The default, when nothing is chosen or the chosen file is gone

For each variant, in order:

1. the **recommended** precision, if it's on disk;
2. otherwise the **highest precision on disk that fits**, using the rule `recommended` already uses: DiT bytes plus 2 GiB of headroom must fit the selected GPU's memory, from `NativeRuntimeInfo::selected_memory_bytes`;
3. otherwise the **smallest on disk**.

This lives in `carey_models.rs` as `active_precision(variant, choices, on_disk, memory) -> Option<tier>`. A unit test covers each branch.

### What ace-server sees

`Service::prepare_launch` already runs just before a native server starts, and writes each published LoRA's `adapter_info.json`. Its signature becomes `-> Result<Vec<(String, String)>, String>`: environment overrides that `start_native` appends to the server's environment.

Carey's `prepare_launch` builds `carey/active-models/`, then returns `ACE_MODELS_DIR=<that folder>`:

- **Contents:** hard links to the shared files (text encoder, VAE) plus one GGUF per variant at its active precision, under their **original file names**, since gary4juce reads the names.
- **Rebuilt on every start:** stale links are removed first, and Carey is stopped while it runs.
- **Fallback:** if a hard link fails, for example when the models folder is on another volume, it logs the reason and returns no override. ace-server then sees the full folder, as today. Copying multi-GB GGUFs is never an option.

Only the public Carey server gets the override:

- **Migration verification** (`native_launch` with an encoding) keeps the full folder, because it proves every downloaded DiT.
- **The captioning server** (`migration_launch`) and **`ace-train`** keep the full folder, because they name exact files themselves; the training dialog has its own precision choice.

### Removing a model

- Removing a non-active precision changes nothing.
- Removing the active one makes the default rule choose again at the next start.
- The 2B base's active precision keeps its current guard: Carey needs it in order to start.

## Panel API

`get_carey_precisions` returns, per variant, `{ active, chosen, on_disk: [tier], fits: { tier: bool } }`, computed in Rust with the default rule above. The panel then needs no logic of its own beyond rendering.

Its row:

```
XL base   [Q8_0 ▾]  active · 4.9 GB on disk · also on disk: Q4_K_M  [use Q4_K_M]
          may run out of GPU memory on this card (needs about 7.1 GB, this GPU has 8.0 GB)
```

## Tests

- **`active_precision`:** recommended on disk; recommended missing so the highest that fits wins; nothing fits so the smallest wins; a stale choice falls back.
- **The `active-models` builder:** only the active files and the shared ones are linked, with the right names; stale links are removed; a failed link falls back to the full folder with no override.
- **`Selection`:** older JSON without `precisions` still parses.
- **Launch:** `start_native` passes the override, and `native_launch` for verification doesn't.
- **A real check** (ignored test or manual): with Q4_K_M and BF16 of XL base on disk and Q4_K_M chosen, `GET /props` lists only `acestep-v15-xl-base-Q4_K_M.gguf`.

## Not in scope

- **Downloading from gary4juce:** selecting a model that isn't on disk. That needs a gary4local listener and a confirm step, and was deferred.
- **A memory check at render time.** The warning is advisory, and an out-of-memory failure still surfaces through `GET /job`'s `error`.
