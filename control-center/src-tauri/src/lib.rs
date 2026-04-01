mod manifest;
mod model_manager;
mod service_manager;

use model_manager::ModelManager;
use service_manager::ServiceManager;
use std::sync::Arc;
use tauri::{Emitter, Manager};
use tauri::menu::{MenuBuilder, MenuItemBuilder};
use tauri::tray::TrayIconBuilder;
use tauri::image::Image;
use tokio::sync::Mutex;

type ManagerState = Arc<Mutex<ServiceManager>>;
type ModelState = Arc<Mutex<ModelManager>>;

/// Load a PNG file and decode it to RGBA for Tauri Image
fn load_png_as_image(path: &std::path::Path) -> Option<Image<'static>> {
    let file = std::fs::File::open(path).ok()?;
    let decoder = png::Decoder::new(file);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());

    let rgba = match info.color_type {
        png::ColorType::Rgba => buf,
        png::ColorType::Rgb => {
            let mut rgba = Vec::with_capacity(buf.len() / 3 * 4);
            for chunk in buf.chunks(3) {
                rgba.extend_from_slice(chunk);
                rgba.push(255);
            }
            rgba
        }
        _ => return None,
    };

    Some(Image::new_owned(rgba, info.width, info.height))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // --- Resolve repo root ---
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()));

            let repo_root = exe_dir
                .as_ref()
                .and_then(|d| d.ancestors().find(|a| a.join("services").is_dir()))
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| std::env::current_dir().unwrap());

            let manifest_path = repo_root.join("services").join("manifests").join("services.json");
            log::info!("Loading manifest from: {}", manifest_path.display());

            let services = match manifest::load_manifest(&manifest_path) {
                Ok(s) => s,
                Err(e) => {
                    log::error!("Failed to load manifest: {}", e);
                    Vec::new()
                }
            };

            let manager = Arc::new(Mutex::new(ServiceManager::new(services, repo_root.clone())));
            app.manage(manager.clone());

            let model_mgr = Arc::new(Mutex::new(ModelManager::new(repo_root.clone())));
            app.manage(model_mgr.clone());

            // Store repo_root for model downloads
            app.manage(repo_root.clone());

            // --- System tray with Gary icon ---
            let handle = app.handle().clone();
            let tray_manager = manager.clone();

            let icon_path = repo_root.join("icon.png");
            let tray_icon = if icon_path.exists() {
                load_png_as_image(&icon_path)
            } else {
                None
            };

            let show_item = MenuItemBuilder::with_id("show", "Show Control Center").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quit").build(app)?;

            let tray_menu = MenuBuilder::new(app)
                .item(&show_item)
                .separator()
                .item(&quit_item)
                .build()?;

            let mut tray_builder = TrayIconBuilder::new()
                .tooltip("gary4juce control center")
                .menu(&tray_menu)
                .on_menu_event(move |app_handle, event| {
                    match event.id().as_ref() {
                        "show" => {
                            if let Some(window) = app_handle.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        "quit" => {
                            let mgr = tray_manager.clone();
                            let handle = app_handle.clone();
                            tauri::async_runtime::spawn(async move {
                                let mut m = mgr.lock().await;
                                m.stop_all();
                                handle.exit(0);
                            });
                        }
                        _ => {}
                    }
                });

            if let Some(icon) = tray_icon {
                tray_builder = tray_builder.icon(icon);
            }

            tray_builder.build(app)?;

            // --- Background: health check + crash detection polling ---
            let poll_manager = manager.clone();
            let poll_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(3))
                    .build()
                    .unwrap();

                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

                    let mut mgr = poll_manager.lock().await;
                    mgr.check_processes();
                    let targets = mgr.get_health_targets();
                    drop(mgr);

                    for (id, port, endpoint) in &targets {
                        let url = format!("http://localhost:{}{}", port, endpoint);
                        let healthy = match client.get(&url).send().await {
                            Ok(resp) => resp.status().is_success(),
                            Err(_) => false,
                        };

                        let mut mgr = poll_manager.lock().await;
                        mgr.set_health(id, healthy);
                    }

                    let mgr = poll_manager.lock().await;
                    let info = mgr.get_service_info();
                    drop(mgr);
                    let _ = poll_handle.emit("services-updated", &info);
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_services,
            start_service,
            stop_service,
            restart_service,
            rebuild_env,
            rebuild_all_envs,
            get_service_log,
            get_models,
            download_model,
            get_download_progress,
            fetch_jerry_checkpoints,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[tauri::command]
async fn get_services(
    manager: tauri::State<'_, ManagerState>,
) -> Result<Vec<service_manager::ServiceInfo>, String> {
    let mut mgr = manager.lock().await;
    mgr.check_processes();
    Ok(mgr.get_service_info())
}

#[tauri::command]
async fn start_service(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<(), String> {
    let mut mgr = manager.lock().await;
    mgr.start(&service_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn stop_service(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<(), String> {
    let mut mgr = manager.lock().await;
    mgr.stop(&service_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn restart_service(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<(), String> {
    let mut mgr = manager.lock().await;
    mgr.stop(&service_id).ok();
    drop(mgr);
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    let mut mgr = manager.lock().await;
    mgr.start(&service_id).map_err(|e| e.to_string())
}

#[tauri::command]
async fn rebuild_env(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let build_info = {
        let mut mgr = manager.lock().await;
        let info = mgr.get_build_info(&service_id)?;
        // +2 for "ensure uv" and "create venv" steps
        let total = info.build_steps.len() + 2;
        mgr.set_build_started(&service_id, total);
        info
    };

    let sid = service_id.clone();
    let mgr_clone = manager.inner().clone();
    let handle = app_handle.clone();

    tauri::async_runtime::spawn(async move {
        let result = run_build(build_info, mgr_clone.clone(), handle.clone()).await;

        let mut mgr = mgr_clone.lock().await;
        match result {
            Ok(()) => mgr.set_build_done(&sid, None),
            Err(e) => mgr.set_build_done(&sid, Some(e)),
        }

        let info = mgr.get_service_info();
        let _ = handle.emit("services-updated", &info);
    });

    Ok(())
}

#[tauri::command]
async fn rebuild_all_envs(
    manager: tauri::State<'_, ManagerState>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let build_infos = {
        let mgr = manager.lock().await;
        mgr.get_all_build_infos()
    };

    for info in build_infos {
        let sid = info.service_id.clone();
        {
            let mut mgr = manager.lock().await;
            let total = info.build_steps.len() + 2;
            mgr.set_build_started(&sid, total);
        }

        let mgr_clone = manager.inner().clone();
        let handle = app_handle.clone();
        let sid_clone = sid.clone();

        tauri::async_runtime::spawn(async move {
            let result = run_build(info, mgr_clone.clone(), handle.clone()).await;

            let mut mgr = mgr_clone.lock().await;
            match result {
                Ok(()) => mgr.set_build_done(&sid_clone, None),
                Err(e) => mgr.set_build_done(&sid_clone, Some(e)),
            }

            let info = mgr.get_service_info();
            let _ = handle.emit("services-updated", &info);
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Build pipeline: uv bootstrap -> Python 3.11 -> venv -> install steps
// ---------------------------------------------------------------------------

/// Ensure uv is installed. Returns the path to the uv executable.
async fn ensure_uv(
    service_id: &str,
    manager: &Arc<Mutex<ServiceManager>>,
    handle: &tauri::AppHandle,
) -> Result<String, String> {
    {
        let mut mgr = manager.lock().await;
        mgr.set_build_step(service_id, 0, "Checking for uv package manager...");
        mgr.append_build_log(service_id, "$ uv --version");
    }
    emit_status(manager, handle).await;

    // Check if uv is already on PATH
    let check = tokio::process::Command::new("uv")
        .arg("--version")
        .output()
        .await;

    if let Ok(output) = &check {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(service_id, &format!("uv found: {}", version.trim()));
            }
            return Ok("uv".to_string());
        }
    }

    // Check common install locations on Windows
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    let cargo_uv = format!("{}\\.cargo\\bin\\uv.exe", home);
    let local_uv = format!("{}\\.local\\bin\\uv.exe", home);

    for path in &[&cargo_uv, &local_uv] {
        if std::path::Path::new(path).exists() {
            let check = tokio::process::Command::new(path)
                .arg("--version")
                .output()
                .await;
            if let Ok(output) = check {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    {
                        let mut mgr = manager.lock().await;
                        mgr.append_build_log(service_id, &format!("uv found at {}: {}", path, version.trim()));
                    }
                    return Ok(path.to_string());
                }
            }
        }
    }

    // Install uv via PowerShell
    {
        let mut mgr = manager.lock().await;
        mgr.append_build_log(service_id, "uv not found. Installing via PowerShell...");
        mgr.append_build_log(service_id, "$ powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"");
    }
    emit_status(manager, handle).await;

    let install = tokio::process::Command::new("powershell")
        .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command",
               "irm https://astral.sh/uv/install.ps1 | iex"])
        .output()
        .await
        .map_err(|e| format!("Failed to run PowerShell uv installer: {}", e))?;

    {
        let mut mgr = manager.lock().await;
        let stdout = String::from_utf8_lossy(&install.stdout);
        let stderr = String::from_utf8_lossy(&install.stderr);
        if !stdout.is_empty() { mgr.append_build_log(service_id, &stdout); }
        if !stderr.is_empty() { mgr.append_build_log(service_id, &stderr); }
    }

    if !install.status.success() {
        return Err("Failed to install uv. Please install manually: https://docs.astral.sh/uv/".to_string());
    }

    // After install, uv should be at ~/.local/bin/uv.exe or ~/.cargo/bin/uv.exe
    for path in &[&local_uv, &cargo_uv] {
        if std::path::Path::new(path).exists() {
            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(service_id, &format!("uv installed at {}", path));
            }
            return Ok(path.to_string());
        }
    }

    // Try PATH again (installer may have updated it)
    let recheck = tokio::process::Command::new("uv")
        .arg("--version")
        .output()
        .await;
    if let Ok(output) = recheck {
        if output.status.success() {
            return Ok("uv".to_string());
        }
    }

    Err("uv was installed but cannot be found. Restart the application and try again.".to_string())
}

/// Run the full build pipeline for a service
async fn run_build(
    build_info: service_manager::BuildInfo,
    manager: Arc<Mutex<ServiceManager>>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    let service_id = build_info.service_id.clone();
    let work_dir = &build_info.work_dir;
    let env_dir = &build_info.env_dir;

    // Step 0: Ensure uv is available
    let uv = ensure_uv(&service_id, &manager, &handle).await?;

    // Step 1: Create venv with uv (using Python 3.11)
    if !env_dir.exists() {
        {
            let mut mgr = manager.lock().await;
            mgr.set_build_step(&service_id, 1, "Installing Python 3.11 and creating venv...");
            mgr.append_build_log(&service_id, &format!("\n$ {} python install 3.11", uv));
        }
        emit_status(&manager, &handle).await;

        // First ensure Python 3.11 is available via uv
        let py_install = run_command_streamed(
            &uv,
            &["python", "install", "3.11"],
            work_dir,
            &service_id,
            &manager,
        ).await;

        if let Err(e) = py_install {
            let mut mgr = manager.lock().await;
            mgr.append_build_log(&service_id, &format!("Warning: uv python install: {}", e));
            // Non-fatal — Python 3.11 might already be on PATH
        }

        // Create the venv
        {
            let mut mgr = manager.lock().await;
            mgr.append_build_log(&service_id, &format!("\n$ {} venv --python 3.11 env", uv));
        }
        emit_status(&manager, &handle).await;

        let venv_output = tokio::process::Command::new(&uv)
            .args(["venv", "--python", "3.11", "env"])
            .current_dir(work_dir)
            .output()
            .await
            .map_err(|e| format!("Failed to create venv: {}", e))?;

        {
            let mut mgr = manager.lock().await;
            let stdout = String::from_utf8_lossy(&venv_output.stdout);
            let stderr = String::from_utf8_lossy(&venv_output.stderr);
            if !stdout.is_empty() { mgr.append_build_log(&service_id, &stdout); }
            if !stderr.is_empty() { mgr.append_build_log(&service_id, &stderr); }
        }

        if !venv_output.status.success() {
            return Err("Failed to create virtual environment with uv".to_string());
        }
    } else {
        let mut mgr = manager.lock().await;
        mgr.set_build_step(&service_id, 1, "Virtual environment already exists");
        mgr.append_build_log(&service_id, "Venv already exists, skipping creation");
    }
    emit_status(&manager, &handle).await;

    let python_exe = env_dir.join("Scripts").join("python.exe");

    // Execute build steps — translate "pip install ..." to "uv pip install ... --python ..."
    for (i, step) in build_info.build_steps.iter().enumerate() {
        let step_num = i + 2; // offset by 2 (uv bootstrap + venv creation)

        // Handle xformers shim specially
        if step == "install_xformers_shim" {
            {
                let mut mgr = manager.lock().await;
                mgr.set_build_step(&service_id, step_num, "Installing xformers SDPA shim...");
                mgr.append_build_log(&service_id, &format!(
                    "\n--- Step {}/{}: install_xformers_shim ---",
                    i + 1, build_info.build_steps.len()
                ));
            }
            emit_status(&manager, &handle).await;

            service_manager::install_xformers_shim(env_dir)?;

            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(&service_id, "xformers SDPA shim installed successfully");
            }
            continue;
        }

        // Translate pip commands to uv pip commands
        let (program, args) = if step.starts_with("pip ") {
            // "pip install ..." -> uv pip install ... --python env/Scripts/python.exe
            let pip_args = &step[4..]; // everything after "pip "
            let mut arg_vec: Vec<String> = vec!["pip".to_string()];
            arg_vec.extend(pip_args.split_whitespace().map(|s| s.to_string()));
            arg_vec.push("--python".to_string());
            arg_vec.push(python_exe.to_str().unwrap().to_string());
            (uv.clone(), arg_vec)
        } else {
            let parts: Vec<&str> = step.splitn(2, ' ').collect();
            let args = parts.get(1).unwrap_or(&"")
                .split_whitespace()
                .map(|s| s.to_string())
                .collect();
            (parts[0].to_string(), args)
        };

        let uv_command_str = format!("{} {}", program, args.join(" "));

        // Truncate step label for UI display
        let short_label = if uv_command_str.len() > 70 {
            format!("{}...", &uv_command_str[..67])
        } else {
            uv_command_str.clone()
        };

        {
            let mut mgr = manager.lock().await;
            mgr.set_build_step(&service_id, step_num, &format!(
                "Step {}/{}: {}",
                i + 1, build_info.build_steps.len(), short_label
            ));
            mgr.append_build_log(&service_id, &format!(
                "\n--- Step {}/{} ---\n$ {}",
                i + 1, build_info.build_steps.len(), uv_command_str
            ));
        }
        emit_status(&manager, &handle).await;

        // Run the command with streamed output
        let mut child = tokio::process::Command::new(&program)
            .args(&args)
            .current_dir(work_dir)
            .env("PYTHONIOENCODING", "utf-8")
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("Build step failed '{}': {}", uv_command_str, e))?;

        // Stream stdout and stderr
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        let mgr_stdout = manager.clone();
        let sid_stdout = service_id.clone();

        let stdout_task = tauri::async_runtime::spawn(async move {
            if let Some(stdout) = stdout {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let mut mgr = mgr_stdout.lock().await;
                    mgr.append_build_log(&sid_stdout, &line);
                }
            }
        });

        let mgr_stderr = manager.clone();
        let sid_stderr = service_id.clone();

        let stderr_task = tauri::async_runtime::spawn(async move {
            if let Some(stderr) = stderr {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut reader = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = reader.next_line().await {
                    let mut mgr = mgr_stderr.lock().await;
                    mgr.append_build_log(&sid_stderr, &line);
                }
            }
        });

        let _ = stdout_task.await;
        let _ = stderr_task.await;

        let exit_status = child.wait().await
            .map_err(|e| format!("Failed to wait for process: {}", e))?;

        if !exit_status.success() {
            let msg = format!("Build step failed (exit code {}): {}",
                exit_status.code().unwrap_or(-1), uv_command_str);
            {
                let mut mgr = manager.lock().await;
                mgr.append_build_log(&service_id, &format!("\nERROR: {}", msg));
            }
            emit_status(&manager, &handle).await;
            return Err(msg);
        }

        {
            let mut mgr = manager.lock().await;
            mgr.append_build_log(&service_id, "OK");
        }
        emit_status(&manager, &handle).await;
    }

    {
        let mut mgr = manager.lock().await;
        mgr.append_build_log(&service_id, &format!("\n=== Build complete for {} ===", service_id));
    }

    Ok(())
}

/// Run a command and stream its output to the build log (non-fatal helper)
async fn run_command_streamed(
    program: &str,
    args: &[&str],
    work_dir: &std::path::Path,
    service_id: &str,
    manager: &Arc<Mutex<ServiceManager>>,
) -> Result<(), String> {
    let output = tokio::process::Command::new(program)
        .args(args)
        .current_dir(work_dir)
        .output()
        .await
        .map_err(|e| format!("Failed to run {} {:?}: {}", program, args, e))?;

    {
        let mut mgr = manager.lock().await;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stdout.is_empty() { mgr.append_build_log(service_id, &stdout); }
        if !stderr.is_empty() { mgr.append_build_log(service_id, &stderr); }
    }

    if output.status.success() {
        Ok(())
    } else {
        Err(format!("Command failed with exit code {}", output.status.code().unwrap_or(-1)))
    }
}

/// Emit current service status to frontend
async fn emit_status(manager: &Arc<Mutex<ServiceManager>>, handle: &tauri::AppHandle) {
    let mgr = manager.lock().await;
    let info = mgr.get_service_info();
    drop(mgr);
    let _ = handle.emit("services-updated", &info);
}

#[tauri::command]
async fn get_service_log(
    service_id: String,
    manager: tauri::State<'_, ManagerState>,
) -> Result<String, String> {
    let mgr = manager.lock().await;
    mgr.get_log(&service_id).map_err(|e| e.to_string())
}

// ---------------------------------------------------------------------------
// Model management commands
// ---------------------------------------------------------------------------

#[tauri::command]
async fn get_models(
    model_mgr: tauri::State<'_, ModelState>,
) -> Result<Vec<model_manager::ModelEntry>, String> {
    let mgr = model_mgr.lock().await;
    let mut models = mgr.get_gary_models();
    models.extend(mgr.get_jerry_models());
    models.extend(mgr.get_carey_models());
    models.extend(mgr.get_foundation_models());
    Ok(models)
}

#[tauri::command]
async fn download_model(
    model_id: String,
    service_id: Option<String>,
    model_mgr: tauri::State<'_, ModelState>,
    _svc_mgr: tauri::State<'_, ManagerState>,
    repo_root: tauri::State<'_, std::path::PathBuf>,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    // Pick the Python env based on which service the model belongs to.
    let svc = service_id.as_deref().unwrap_or("gary");
    let env_dir = match svc {
        "stable-audio" => "stable-audio",
        "carey" => "carey",
        "foundation" => "foundation",
        _ => "gary",
    };
    let python_exe = repo_root.join("services").join(env_dir).join("env")
        .join("Scripts").join("python.exe");

    if !python_exe.exists() {
        let label = match env_dir {
            "stable-audio" => "Jerry (Stable Audio)",
            "carey" => "Carey (ACE-Step)",
            "foundation" => "Foundation-1",
            _ => "Gary",
        };
        return Err(format!("{}'s Python environment is not built yet. Build it first.", label));
    }

    {
        let mut mgr = model_mgr.lock().await;
        mgr.set_download_started(&model_id);
    }

    let mgr_clone = model_mgr.inner().clone();
    let handle = app_handle.clone();
    let root: std::path::PathBuf = repo_root.to_path_buf();

    // Carey models download to checkpoints/ dir, not HF cache
    if model_id.starts_with("carey::") {
        let checkpoint_dir = root.join("services").join("carey").join("checkpoints");
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_carey_model(
                model_id, python_exe, checkpoint_dir, mgr_clone, handle,
            ).await;
        });
    } else if model_id.starts_with("foundation::") {
        // Foundation models go to %APPDATA%/Gary4JUCE/models/foundation-1/
        let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
            let home = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".to_string());
            format!("{}\\AppData\\Roaming", home)
        });
        let model_dir = std::path::PathBuf::from(appdata)
            .join("Gary4JUCE").join("models").join("foundation-1");
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_foundation_model(
                model_id, python_exe, model_dir, mgr_clone, handle,
            ).await;
        });
    } else {
        tauri::async_runtime::spawn(async move {
            let _ = model_manager::download_model(
                model_id, python_exe, mgr_clone, handle,
            ).await;
        });
    }

    Ok(())
}

#[tauri::command]
async fn fetch_jerry_checkpoints(
    repo: String,
    model_mgr: tauri::State<'_, ModelState>,
    app_handle: tauri::AppHandle,
) -> Result<Vec<String>, String> {
    // Jerry runs on port 8005
    let checkpoints = model_manager::fetch_checkpoints(repo.clone(), 8005).await?;

    // Store in model manager so they appear in the model list
    {
        let mut mgr = model_mgr.lock().await;
        mgr.set_finetune_checkpoints(&repo, checkpoints.clone());
    }

    // Emit updated model list
    let mgr_arc = model_mgr.inner().clone();
    model_manager::emit_model_status(&mgr_arc, &app_handle).await;

    Ok(checkpoints)
}

#[tauri::command]
async fn get_download_progress(
    model_mgr: tauri::State<'_, ModelState>,
) -> Result<Vec<model_manager::DownloadProgress>, String> {
    let mgr = model_mgr.lock().await;
    Ok(mgr.get_download_progress())
}
