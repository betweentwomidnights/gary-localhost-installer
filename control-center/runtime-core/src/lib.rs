//! Runtime building blocks shared by the desktop app and GUI-free supervisor.
pub mod download;
pub mod cli;
pub mod command;
pub mod console;
pub mod headless;
pub mod manifest;
pub mod model_catalog;
pub mod runtime;
pub mod sa3_adapter;
pub mod workload_job;

pub mod service;

pub mod ace_client;
pub mod carey_caption;
pub mod carey_dataset;
pub mod dataset;
pub mod sa3_gguf;
pub mod training;

mod training_job;
