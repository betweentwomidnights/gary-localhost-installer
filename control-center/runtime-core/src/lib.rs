//! Runtime building blocks shared by the desktop app and GUI-free supervisor.
pub mod download;
pub mod cli;
pub mod command;
pub mod console;
pub mod headless;
pub mod listener;
pub mod manifest;
pub mod model_catalog;
pub mod model_store;
pub mod runtime;
pub mod sa3_adapter;
pub mod workload_job;

pub mod service;

pub mod ace_client;
pub mod carey_adapter;
pub mod carey_caption;
pub mod carey_dataset;
pub mod dataset;
pub mod sa3_gguf;
pub mod sa3_prompts;
pub mod training;
pub mod lora_library;
pub mod storage_paths;

mod training_job;
mod lora_history;
