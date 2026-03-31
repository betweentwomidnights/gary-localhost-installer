/// <reference types="svelte" />
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ENABLE_MELODYFLOW_FA2_TOGGLE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
