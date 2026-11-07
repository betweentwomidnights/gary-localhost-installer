// How the Models panel lays out a native service whose server keeps every
// downloaded model and is told per request which one to run, so each model is
// its own choice of precision. One model (`required`) is what the service
// needs to start, at the precision its migration selected.

export interface VariantGroup {
  label: string;
  hint: string;
  variants: string[];
}

export interface VariantLayout {
  // The catalog's component prefix: `<prefix>::shared`, `<prefix>::<variant>::<tier>`.
  prefix: string;
  label: string;
  required: string;
  // How the copy names the required model.
  requiredLabel: string;
  groups: VariantGroup[];
}

export const careyLayout: VariantLayout = {
  prefix: "carey-native",
  label: "Carey",
  required: "base",
  requiredLabel: "the 2B base",
  groups: [
    { label: "2B models", hint: "The base model runs every task, including lego, extract and complete, and Carey needs it to start. sft and turbo are optional.", variants: ["base", "sft", "turbo"] },
    { label: "XL models", hint: "Optional. XL is larger and needs more GPU memory than 2B at the same precision. Some tasks, such as lego, can sound better on 2B.", variants: ["xl-base", "xl-sft", "xl-turbo"] },
    { label: "language models", hint: "Optional. A language model writes lyrics, metadata and audio codes for standard generation; Carey's DAW tasks run without one.", variants: ["lm-0.6B", "lm-1.7B", "lm-4B"] },
  ],
};

// musicgen-server lists every model on disk in gary4juce's dropdown and runs
// the most precise copy of each, so a second precision only takes space.
export const garyLayout: VariantLayout = {
  prefix: "gary-native",
  label: "gary",
  required: "vanya_ai_dnb_0.1",
  requiredLabel: "vanya_ai_dnb_0.1",
  groups: [
    { label: "finetunes", hint: "thepatch's finetunes. gary needs vanya_ai_dnb_0.1 to start, and gary4juce falls back to it; the others are optional. Each one downloaded appears in gary4juce's model dropdown. If a model is on disk at more than one precision, gary runs the most precise.", variants: ["vanya_ai_dnb_0.1", "hoenn_lofi", "keygen-gary-v2-large-12"] },
    { label: "base models", hint: "Optional. Meta's original MusicGen, listed in gary4juce after the finetunes of the same size. Larger models need more GPU memory at the same precision.", variants: ["musicgen-small", "musicgen-medium", "musicgen-large"] },
  ],
};

// By service id.
export const variantLayouts: Record<string, VariantLayout> = { carey: careyLayout, gary: garyLayout };
