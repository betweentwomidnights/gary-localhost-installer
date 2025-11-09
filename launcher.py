import subprocess
import time
import sys
import os
from pathlib import Path
import threading
import requests
from dotenv import dotenv_values
from queue import Queue, Empty
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QFrame, QGridLayout, QSplitter,
    QSystemTrayIcon, QMenu, QMessageBox, QGroupBox, QProgressBar,
    QTabWidget, QScrollArea
)
from PySide6.QtCore import QTimer, QThread, Signal, Qt, QSize
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QAction
import psutil
try:
    import pynvml
    pynvml.nvmlInit()
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
except Exception:
    NVIDIA_AVAILABLE = False

class ServiceWorker(QThread):
    """Worker thread for service management operations."""
    log_message = Signal(str, str)  # service_name, message
    status_changed = Signal()
    services_started = Signal()
    services_stopped = Signal()
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.operation = None
        self.running = True
        
    def start_services(self):
        """Queue start services operation."""
        self.operation = "start"
        self.start()
        
    def stop_services(self):
        """Queue stop services operation."""
        self.operation = "stop"
        self.start()
        
    def run(self):
        """Execute the queued operation."""
        if self.operation == "start":
            self._start_services()
        elif self.operation == "stop":
            self._stop_services()
            
    def _start_services(self):
        """Start all services (runs in background thread)."""
        if self.parent.services_running:
            return
            
        self.log_message.emit("system", "🚀 starting gary4juce backend...")
        self.parent.services_running = True
        self.parent.start_time = time.time()
        self.status_changed.emit()
        
        try:
            # Start Redis (keep existing Redis code unchanged)
            redis_exe = self.parent.base_dir / "redis" / "redis-server.exe"
            if not redis_exe.exists():
                self.log_message.emit("system", "❌ Redis not found - please run installer first")
                return

            # Now try to start Redis
            self.log_message.emit("system", "▶️ Starting Redis server...")
            try:
                redis_cwd = self.parent.base_dir / "redis"
                
                # Use default Redis configuration (no config file)
                # This matches what happens when you double-click redis-server.exe
                redis_cmd = [str(redis_exe)]
                
                self.log_message.emit("system", "📋 Using Redis default configuration")
                
                # Start Redis process
                redis_process = subprocess.Popen(
                    redis_cmd,
                    cwd=str(redis_cwd),
                    stdout=subprocess.DEVNULL,  # Suppress Redis output
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    text=True
                )
                
                self.parent.processes["redis"] = redis_process
                self.log_message.emit("system", f"✅ Redis process started (PID: {redis_process.pid})")
                
                # Wait a moment and verify Redis is responding
                time.sleep(3)
                
                # Check if Redis process is still alive
                if redis_process.poll() is not None:
                    exit_code = redis_process.poll()
                    self.log_message.emit("system", f"❌ Redis process died immediately (exit code: {exit_code})")
                    self.log_message.emit("system", f"💡 Try manually deleting: {self.parent.base_dir / 'redis' / 'redis.windows-service.conf'}")
                    return
                
                # Test Redis connection
                max_retries = 5
                for attempt in range(max_retries):
                    if self.parent.check_redis_health():
                        self.log_message.emit("system", "✅ Redis server is responding")
                        break
                    elif attempt < max_retries - 1:
                        self.log_message.emit("system", f"⏳ Waiting for Redis... ({attempt + 1}/{max_retries})")
                        time.sleep(2)
                    else:
                        self.log_message.emit("system", "⚠️ Redis started but not responding to health checks")
                
            except Exception as e:
                self.log_message.emit("system", f"❌ Failed to start Redis: {e}")
                return
            
            # Load environment variables
            env_vars = os.environ.copy()
            env_file = self.parent.base_dir / ".env"
            if env_file.exists():
                env_vars.update(dotenv_values(str(env_file)))
            
            # Start each service with DEBUG OUTPUT
            for name, config in self.parent.services.items():
                python_exe = config["dir"] / "env" / "Scripts" / "python.exe"
                script_path = config["dir"] / config["script"]
                
                if not python_exe.exists() or not script_path.exists():
                    self.log_message.emit("system", f"❌ {config['name']} files not found - Python: {python_exe.exists()}, Script: {script_path.exists()}")
                    continue
                
                self.log_message.emit("system", f"▶️ starting {config['name']} on port {config['port']}...")
                
                service_env = env_vars.copy()
                if name == "melodyflow":
                    service_env["XFORMERS_DISABLED"] = "1"
                    service_env["FLASH_ATTENTION_DISABLED"] = "1"
                
                # Fix Unicode encoding issues on Windows
                service_env["PYTHONIOENCODING"] = "utf-8"
                service_env["PYTHONLEGACYWINDOWSSTDIO"] = "1"
                
                try:
                    # TEMPORARY DEBUG: Capture output to see errors
                    log_file = self.parent.base_dir / f"{name}_debug.log"
                    
                    with open(log_file, 'w', encoding='utf-8') as log:
                        process = subprocess.Popen(
                            [str(python_exe), str(script_path)],
                            cwd=str(config["dir"]),
                            env=service_env,
                            stdout=log,           # Capture stdout to file
                            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                            creationflags=subprocess.CREATE_NO_WINDOW,
                            text=True
                        )
                    
                    self.parent.processes[name] = process
                    self.log_message.emit("system", f"✅ {config['name']} process started (PID: {process.pid})")
                    self.log_message.emit("system", f"   Debug log: {log_file}")
                    
                    # Give service a moment, then check if it's still alive
                    time.sleep(3)  # Increased wait time
                    
                    if process.poll() is not None:
                        # Process already died - read the log file
                        exit_code = process.poll()
                        self.log_message.emit("system", f"❌ {config['name']} process died immediately (exit code: {exit_code})")
                        
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                error_output = f.read()
                            if error_output.strip():
                                # Log first few lines of error
                                error_lines = error_output.strip().split('\n')[:5]
                                for line in error_lines:
                                    self.log_message.emit("system", f"   ERROR: {line}")
                            else:
                                self.log_message.emit("system", f"   No error output captured")
                        except Exception as e:
                            self.log_message.emit("system", f"   Could not read debug log: {e}")
                            
                        # Special handling for Windows Access Violation
                        if exit_code == 3221225477:
                            self.log_message.emit("system", f"   Windows Access Violation detected - check dependencies and antivirus")
                    else:
                        self.log_message.emit("system", f"   {config['name']} process running successfully")
                        
                except Exception as e:
                    self.log_message.emit("system", f"❌ Failed to start {config['name']}: {e}")
                    
            self.log_message.emit("system", "✅ all services started!")
            self.log_message.emit("system", "⏳ services may take a moment to become healthy...")
            self.log_message.emit("system", "📝 Check debug logs if services fail to start:")
            for name in self.parent.services.keys():
                log_file = self.parent.base_dir / f"{name}_debug.log"
                self.log_message.emit("system", f"   {log_file}")
            self.services_started.emit()
            
        except Exception as e:
            self.log_message.emit("system", f"❌ error starting services: {e}")
            
    def _stop_services(self):
        """Stop all services (runs in background thread)."""
        if not self.parent.services_running:
            return
            
        self.log_message.emit("system", "⏹️ stopping gary4juce services...")
        
        # Stop processes
        for name, process in self.parent.processes.items():
            if process and process.poll() is None:
                try:
                    config_name = self.parent.services.get(name, {}).get("name", name)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                        self.log_message.emit("system", f"🛑 stopped {config_name}")
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                        self.log_message.emit("system", f"🛑 force stopped {config_name}")
                except Exception as e:
                    self.log_message.emit("system", f"❌ error stopping {name}: {e}")
        
        # Force kill remaining processes
        try:
            subprocess.run(["taskkill", "/F", "/IM", "redis-server.exe"], 
                         capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            self.log_message.emit("system", "🛑 stopped redis processes")
        except:
            pass
                    
        self.parent.processes.clear()
        self.parent.services_running = False
        self.parent.start_time = None  # Reset start time
        self.log_message.emit("system", "✅ all services stopped.")
        self.services_stopped.emit()

class HealthChecker(QThread):
    """Background thread for smart health checking."""
    health_status = Signal(dict)  # status dict for all services
    detailed_health = Signal(str, dict)  # service_name, detailed_status
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True
        self.detailed_check_counter = 0
        self.services_confirmed_healthy = {}  # Track which services are confirmed healthy
        self.last_full_health_check = 0
        
    def run(self):
        """Smart health checking with adaptive frequency."""
        while self.running:
            try:
                status = {}
                current_time = time.time()
                
                # Check Redis (lightweight, always check)
                status['redis'] = self.parent.check_redis_health()
                
                # Smart service checking
                for name, config in self.parent.services.items():
                    if self.parent.services_running and name in self.parent.processes:
                        process = self.parent.processes[name]
                        if process and process.poll() is None:  # Process is running
                            
                            # If service was confirmed healthy recently, use lighter checks
                            if (name in self.services_confirmed_healthy and 
                                current_time - self.services_confirmed_healthy[name] < 300):  # 5 minutes
                                
                                # Just check if port is still listening (much lighter)
                                status[name] = self.parent.check_port_listening(config["port"])
                                
                            else:
                                # Do full HTTP health check for new/unconfirmed services
                                timeout = 10 if name == "gary" else 3
                                
                                # Occasional detailed check for troubleshooting
                                if self.detailed_check_counter % 20 == 0:  # Every 20th check
                                    detailed = self.parent.check_service_health_detailed(config["port"], timeout=timeout)
                                    self.detailed_health.emit(name, detailed)
                                    health_ok = detailed["healthy"]
                                else:
                                    health_ok = self.parent.check_service_health(config["port"], timeout=timeout)
                                
                                if health_ok:
                                    # Mark as confirmed healthy
                                    self.services_confirmed_healthy[name] = current_time
                                    status[name] = True
                                    
                                    # Log confirmation only once
                                    if name not in self.services_confirmed_healthy or current_time - self.services_confirmed_healthy.get(name, 0) > 300:
                                        self.parent.add_log("system", f"✅ {config['name']} is healthy")
                                else:
                                    # Remove from confirmed healthy if check fails
                                    if name in self.services_confirmed_healthy:
                                        del self.services_confirmed_healthy[name]
                                        self.parent.add_log("system", f"⚠️ {config['name']} health check failed")
                                    
                                    # Fallback to port check
                                    status[name] = self.parent.check_port_listening(config["port"])
                        else:
                            status[name] = False  # Process died
                            # Remove from confirmed healthy
                            if name in self.services_confirmed_healthy:
                                del self.services_confirmed_healthy[name]
                            # Log if process died unexpectedly
                            if process:
                                exit_code = process.poll()
                                config_name = config.get("name", name)
                                self.parent.add_log("system", f"❌ {config_name} process died (exit code: {exit_code})")
                                if exit_code == 3221225477:  # Windows Access Violation
                                    self.parent.add_log("system", f"   Windows Access Violation - likely dependency issue")
                    else:
                        status[name] = False  # Not supposed to be running
                        # Clear confirmed healthy status
                        if name in self.services_confirmed_healthy:
                            del self.services_confirmed_healthy[name]
                
                self.detailed_check_counter += 1
                self.health_status.emit(status)
                
                # Adaptive sleep timing
                all_services_healthy = all(status.get(name, False) for name in self.parent.services.keys())
                
                if all_services_healthy and len(self.services_confirmed_healthy) == len(self.parent.services):
                    # All services confirmed healthy - check less frequently
                    sleep_time = 15  # 15 seconds
                elif any(status.get(name, False) for name in self.parent.services.keys()):
                    # Some services healthy - moderate frequency  
                    sleep_time = 8   # 8 seconds
                else:
                    # Services starting/unhealthy - check more frequently
                    sleep_time = 3   # 3 seconds
                
                # Sleep in small increments to be responsive to stop
                for _ in range(sleep_time * 10):  # 0.1 second increments
                    if not self.running:
                        break
                    time.sleep(0.1)
                    
            except Exception as e:
                # Log health check errors
                if self.parent:
                    self.parent.add_log("system", f"Health check error: {e}")
                break
                
    def stop(self):
        self.running = False

class ServiceStatusWidget(QWidget):
    """Widget showing status of a single service."""
    
    def __init__(self, service_name, service_config):
        super().__init__()
        self.service_name = service_name
        self.service_config = service_config
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(12, 8, 12, 8)  # Add some padding
        
        # Status indicator (colored circle)
        self.status_label = QLabel("●")
        self.status_label.setStyleSheet("color: #FF4444; font-size: 18px; font-weight: bold;")  # Red by default, larger
        
        # Service name
        name_label = QLabel(self.service_config["name"])
        name_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        name_label.setStyleSheet("color: #FFFFFF;")
        
        # Port info
        port_label = QLabel(f"Port {self.service_config['port']}")
        port_label.setStyleSheet("color: #AAAAAA; font-size: 10px;")
        
        # Add to layout
        layout.addWidget(self.status_label)
        layout.addWidget(name_label)
        layout.addStretch()
        layout.addWidget(port_label)
        
        self.setLayout(layout)
        
    def update_status(self, status):
        """Update visual status indicator."""
        colors = {
            "stopped": "#FF4444",      # Red (matching plugin red accent)
            "starting": "#FFAA00",     # Orange/Yellow
            "running": "#44FF44",      # Green
            "error": "#FF8844"         # Orange
        }
        
        color = colors.get(status, "#FF4444")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")

class Gary4JUCEControlCenter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Program directory (where executable is) - for icons, redis, etc.
        self.base_dir = self.get_base_directory()
        
        # Services directory (in user AppData, same as installer)
        self.services_dir = self.get_services_directory()
        
        self.services_running = False
        self.processes = {}
        self.start_time = None
        self.last_health_status = {}
        
        # Debug log to verify directories
        print(f"DEBUG: Program directory: {self.base_dir}")
        print(f"DEBUG: Services directory: {self.services_dir}")
        print(f"DEBUG: Services directory exists: {self.services_dir.exists()}")
        
        # Load application icon (from program directory)
        self.load_application_icon()
        
        # Service configuration - updated to use services_dir
        self.services = {
            "gary": {
                "dir": self.services_dir / "gary",
                "script": "g4l_localhost.py", 
                "port": 8000,
                "name": "gary main"
            },
            "melodyflow": {
                "dir": self.services_dir / "melodyflow",
                "script": "localhost_melodyflow.py", 
                "port": 8002,
                "name": "terry (melodyflow)"
            },
            "stable-audio": {
                "dir": self.services_dir / "stable-audio",
                "script": "api.py", 
                "port": 8005,
                "name": "jerry (stable audio open small)"
            }
        }
        
        # Debug individual service directories
        for name, config in self.services.items():
            service_dir = config["dir"]
            python_exe = service_dir / "env" / "Scripts" / "python.exe"
            script_file = service_dir / config["script"]
            print(f"DEBUG: {name} dir exists: {service_dir.exists()}")
            print(f"DEBUG: {name} python exists: {python_exe.exists()}")
            print(f"DEBUG: {name} script exists: {script_file.exists()}")
        
        # Initialize worker threads
        self.service_worker = ServiceWorker(self)
        self.service_worker.log_message.connect(self.add_log)
        self.service_worker.status_changed.connect(self.update_status)
        self.service_worker.services_started.connect(self.on_services_started)
        self.service_worker.services_stopped.connect(self.on_services_stopped)
        
        self.health_checker = HealthChecker(self)
        self.health_checker.health_status.connect(self.on_health_status_update)
        self.health_checker.detailed_health.connect(self.on_detailed_health_update)
        self.health_checker.start()  # Start health checking immediately
        
        self.init_ui()
        self.init_tray()
        self.init_timers()
        
    def init_ui(self):
        """Initialize the main window UI."""
        self.setWindowTitle("gary4juce control center")
        self.setGeometry(100, 100, 900, 650)  # Slightly larger window
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Header with title and main controls
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Service Status and Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Compact Status Log
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (more emphasis on status, less on logs)
        splitter.setSizes([500, 400])
        main_layout.addWidget(splitter)
        
        central_widget.setLayout(main_layout)
        
        # Apply modern dark styling
        self.apply_styling()

    

    def get_base_directory(self):
        """Get the correct base directory whether running as script or executable."""
        if getattr(sys, 'frozen', False):
            # Running as packaged executable
            base_dir = Path(sys.executable).parent
        else:
            # Running as script
            base_dir = Path(__file__).parent
        
        return base_dir
    
    def get_services_directory(self):
        """Get the services directory (in user AppData, same as installer)."""
        import os
        if os.name == 'nt':  # Windows
            user_data_dir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
            services_dir = user_data_dir / "Gary4JUCE" / "services"
        else:
            # For other platforms
            services_dir = Path.home() / ".gary4juce" / "services"
        
        return services_dir

    def load_application_icon(self):
        """Load and set the application icon."""
        # Try different icon formats in order of preference (from program directory)
        icon_files = [
            self.base_dir / "icon.ico",  # Windows ICO format (best for Windows)
            self.base_dir / "icon.png",  # PNG fallback
        ]
        
        icon_loaded = False
        for icon_path in icon_files:
            if icon_path.exists():
                try:
                    # Create QIcon from file
                    app_icon = QIcon(str(icon_path))
                    
                    # Set application-wide icon (affects taskbar)
                    QApplication.instance().setWindowIcon(app_icon)
                    
                    # Set window icon (affects window title bar)
                    self.setWindowIcon(app_icon)
                    
                    # Log success
                    if hasattr(self, 'log_viewer'):
                        self.add_log("system", f"✅ Loaded application icon: {icon_path.name}")
                    icon_loaded = True
                    break
                    
                except Exception as e:
                    # Log error
                    if hasattr(self, 'log_viewer'):
                        self.add_log("system", f"⚠️ Failed to load icon {icon_path.name}: {e}")
                    continue
        
        if not icon_loaded:
            # Log warning
            if hasattr(self, 'log_viewer'):
                self.add_log("system", "⚠️ No application icon found - using default")
            # Optionally create a simple fallback icon
            self.create_fallback_icon()
    
    def create_fallback_icon(self):
        """Create a simple fallback icon if no icon file is found."""
        # Create a 32x32 pixmap with our brand colors
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background circle
        painter.setBrush(QColor("#BB0000"))  # Your brand red
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 28, 28)
        
        # Draw text
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "G4J")
        
        painter.end()
        
        # Set as application icon
        fallback_icon = QIcon(pixmap)
        QApplication.instance().setWindowIcon(fallback_icon)
        self.setWindowIcon(fallback_icon)
        
    def create_header(self):
        """Create the header with title and main controls."""
        header = QFrame()
        header.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)
        
        # Title
        title = QLabel("gary4juce control center")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #FFFFFF; margin-right: 20px;")
        layout.addWidget(title)
        
        layout.addStretch()
        
        # Main control buttons
        self.start_btn = QPushButton("start all services")
        self.start_btn.clicked.connect(self.start_services)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(160)
        
        self.stop_btn = QPushButton("stop all services")
        self.stop_btn.clicked.connect(self.stop_services)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setMinimumWidth(160)
        self.stop_btn.setEnabled(False)
        
        layout.addWidget(self.start_btn)
        layout.addSpacing(10)
        layout.addWidget(self.stop_btn)
        
        header.setLayout(layout)
        return header
        
    def create_left_panel(self):
        """Create the left panel with service status and controls."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Service Status Group
        status_group = QGroupBox("service status")
        status_layout = QVBoxLayout()
        status_layout.setSpacing(5)
        
        # Redis status
        self.redis_status = ServiceStatusWidget("redis", {"name": "redis server", "port": 6379})
        status_layout.addWidget(self.redis_status)
        
        # Service status widgets
        self.service_widgets = {}
        for name, config in self.services.items():
            widget = ServiceStatusWidget(name, config)
            self.service_widgets[name] = widget
            status_layout.addWidget(widget)
            
        status_layout.addStretch()
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Individual service controls
        controls_group = QGroupBox("individual controls")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(8)
        
        for name, config in self.services.items():
            service_frame = QFrame()
            service_frame.setStyleSheet("""
                QFrame {
                    background-color: #2A2A2A;
                    border: 1px solid #444444;
                    border-radius: 6px;
                    padding: 8px;
                }
            """)
            service_layout = QHBoxLayout()
            service_layout.setContentsMargins(12, 8, 12, 8)
            
            label = QLabel(config["name"])
            label.setStyleSheet("color: #FFFFFF; font-weight: bold;")
            
            # Stop button - NEW!
            stop_btn = QPushButton("stop")
            stop_btn.clicked.connect(lambda checked, svc=name: self.stop_service(svc))
            stop_btn.setMinimumWidth(80)
            stop_btn.setMinimumHeight(30)
            
            # Restart button
            restart_btn = QPushButton("restart")
            restart_btn.clicked.connect(lambda checked, svc=name: self.restart_service(svc))
            restart_btn.setMinimumWidth(80)
            restart_btn.setMinimumHeight(30)
            
            service_layout.addWidget(label)
            service_layout.addStretch()
            service_layout.addWidget(stop_btn)
            service_layout.addSpacing(8)  # Space between buttons
            service_layout.addWidget(restart_btn)
            
            service_frame.setLayout(service_layout)
            controls_layout.addWidget(service_frame)
            
        controls_layout.addStretch()
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # System info
        info_group = QGroupBox("system info")
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)
        
        self.uptime_label = QLabel("uptime: not running")
        self.uptime_label.setStyleSheet("color: #FFFFFF; font-size: 11px;")
        
        self.cpu_label = QLabel("CPU: --")
        self.cpu_label.setStyleSheet("color: #FFFFFF; font-size: 11px;")
        
        self.memory_label = QLabel("memory: --")
        self.memory_label.setStyleSheet("color: #FFFFFF; font-size: 11px;")
        
        info_layout.addWidget(self.uptime_label)
        info_layout.addWidget(self.cpu_label)
        info_layout.addWidget(self.memory_label)
        info_layout.addStretch()
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        panel.setLayout(layout)
        return panel
        
    def create_right_panel(self):
        """Create the right panel with compact status log."""
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Log viewer header
        log_header = QFrame()
        log_header_layout = QHBoxLayout()
        log_header_layout.setContentsMargins(5, 5, 5, 5)
        
        log_title = QLabel("system status")
        log_title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        log_title.setStyleSheet("color: #FFFFFF;")
        
        clear_btn = QPushButton("clear")
        clear_btn.clicked.connect(self.clear_logs)
        clear_btn.setMinimumWidth(80)
        clear_btn.setMinimumHeight(30)
        
        log_header_layout.addWidget(log_title)
        log_header_layout.addStretch()
        log_header_layout.addWidget(clear_btn)
        
        log_header.setLayout(log_header_layout)
        layout.addWidget(log_header)
        
        # Compact log text area
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setFont(QFont("Consolas", 9))  # Monospace font
        self.log_viewer.setMaximumHeight(250)  # Controlled height
        self.log_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #1A1A1A;
                color: #FFFFFF;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 10px;
                selection-background-color: #FF4444;
            }
        """)
        
        layout.addWidget(self.log_viewer)
        
        # Health status summary
        health_group = QGroupBox("latest health check")
        health_layout = QVBoxLayout()
        health_layout.setContentsMargins(10, 15, 10, 10)
        
        self.health_summary = QLabel("no health check yet")
        self.health_summary.setFont(QFont("Segoe UI", 10))
        self.health_summary.setWordWrap(True)
        self.health_summary.setStyleSheet("color: #FFFFFF; padding: 5px;")
        
        health_layout.addWidget(self.health_summary)
        health_group.setLayout(health_layout)
        layout.addWidget(health_group)
        
        layout.addStretch()
        
        panel.setLayout(layout)
        return panel
        
    def init_tray(self):
        """Initialize system tray icon."""
        # Create tray icon
        self.tray_icon = QSystemTrayIcon(self)
        
        # Create icon (simple colored circle for now)
        self.update_tray_icon("stopped")
        
        # Create tray menu
        self.update_tray_menu()
        
        # Show tray icon
        self.tray_icon.show()
        
        # Handle tray icon activation
        self.tray_icon.activated.connect(self.on_tray_activated)
        
    def create_tray_icon(self, status="stopped"):
        """Create system tray icon based on status."""
        colors = {
            "stopped": QColor("#FF4444"),      # Red (matching plugin)
            "starting": QColor("#FFAA00"),     # Orange
            "running": QColor("#44FF44"),      # Green
            "error": QColor("#FF8844")         # Orange
        }
        
        color = colors.get(status, QColor("#FF4444"))
        
        # Create 32x32 icon
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw circle
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 24, 24)
        
        # Draw text
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 8, QFont.Weight.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "G4J")
        
        painter.end()
        
        return QIcon(pixmap)
        
    def update_tray_icon(self, status):
        """Update tray icon based on status."""
        icon = self.create_tray_icon(status)
        self.tray_icon.setIcon(icon)
        
        # Update tooltip
        status_text = {
            "stopped": "gary4juce - stopped",
            "starting": "gary4juce - starting...",
            "running": "gary4juce - all services running",
            "error": "gary4juce - service error"
        }
        
        self.tray_icon.setToolTip(status_text.get(status, "gary4juce"))
        
    def update_tray_menu(self):
        """Update system tray context menu."""
        menu = QMenu()
        
        # Show/Hide window
        show_action = QAction("show control center", self)
        show_action.triggered.connect(self.show_window)
        menu.addAction(show_action)
        
        menu.addSeparator()
        
        # Service controls
        if self.services_running:
            start_action = QAction("services running", self)
            start_action.setEnabled(False)
        else:
            start_action = QAction("start services", self)
            start_action.triggered.connect(self.start_services)
        menu.addAction(start_action)
        
        stop_action = QAction("stop services", self)
        stop_action.triggered.connect(self.stop_services)
        stop_action.setEnabled(self.services_running)
        menu.addAction(stop_action)
        
        menu.addSeparator()
        
        # Simple exit - always stops services if running
        if self.services_running:
            exit_action = QAction("❌ stop and exit", self)
        else:
            exit_action = QAction("❌ exit", self)
        exit_action.triggered.connect(self.quit_application)
        menu.addAction(exit_action)
        
        self.tray_icon.setContextMenu(menu)
        
    def on_tray_activated(self, reason):
        """Handle tray icon activation."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_window()
            
    def show_window(self):
        """Show and raise the main window."""
        self.show()
        self.raise_()
        self.activateWindow()
        
    def init_timers(self):
        """Initialize update timers."""
        # Lighter weight timer just for UI updates
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.update_ui_only)
        self.ui_timer.start(1000)  # Update UI every 1 second
        
        # Tray menu update timer (less frequent)
        self.menu_timer = QTimer()
        self.menu_timer.timeout.connect(self.update_tray_menu)
        self.menu_timer.start(5000)  # Update every 5 seconds
        
    def on_health_status_update(self, status):
        """Handle health status updates from background thread."""
        self.last_health_status = status
        self.update_status_indicators()
        self.update_health_summary(status)
        
    def update_health_summary(self, status):
        """Update the health summary display."""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Create summary
        summary_parts = []
        for name, healthy in status.items():
            if name == "redis":
                display_name = "Redis"
            else:
                display_name = self.services[name]["name"]
            
            icon = "✅" if healthy else "❌"
            summary_parts.append(f"{icon} {display_name}")
        
        summary_text = f"[{current_time}] " + " | ".join(summary_parts)
        self.health_summary.setText(summary_text)
        
    def on_detailed_health_update(self, service_name, details):
        """Handle detailed health check results - minimal logging."""
        # Only log significant issues
        if not details["success"]:
            if service_name == "gary" and "Timeout" in details.get("error", ""):
                # Don't log gary timeouts as they're usually just busy generating
                pass
            else:
                config = self.services.get(service_name, {})
                service_display_name = config.get("name", service_name)
                self.add_log("system", f"⚠️ {service_display_name}: {details['error']}")
        
    def on_services_started(self):
        """Called when services have finished starting."""
        self.update_status()
        
    def on_services_stopped(self):
        """Called when services have finished stopping."""
        self.update_status()
    
    def get_system_stats(self):
        """Get current system statistics."""
        stats = {}
        
        try:
            # CPU usage
            stats['cpu'] = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            stats['memory_percent'] = memory.percent
            stats['memory_used'] = memory.used // (1024**3)  # GB
            stats['memory_total'] = memory.total // (1024**3)  # GB
            
            # GPU stats (if available)
            if NVIDIA_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    stats['gpu_util'] = gpu_util.gpu
                    stats['vram_used'] = memory_info.used // (1024**2)  # MB
                    stats['vram_total'] = memory_info.total // (1024**2)  # MB
                except Exception:
                    stats['gpu_util'] = None
                    
        except Exception as e:
            # Fallback values if monitoring fails
            stats = {'cpu': None, 'memory_percent': None, 'gpu_util': None}
            
        return stats
        
    def update_ui_only(self):
        """Lightweight UI updates that don't block."""
        # Update tray icon based on current status
        system_status = self.get_system_status()
        self.update_tray_icon(system_status)
        
        # Update system info every few cycles (less frequent)
        if not hasattr(self, '_ui_counter'):
            self._ui_counter = 0
        self._ui_counter += 1
        
        # Update system stats every 3 seconds (every 3rd call since timer is 1 second)
        if self._ui_counter % 3 == 0:
            self.update_system_info()

    def update_system_info(self):
        """Update system information display."""
        stats = self.get_system_stats()
        
        # Update uptime
        if self.start_time and self.services_running:
            uptime_seconds = int(time.time() - self.start_time)
            hours, remainder = divmod(uptime_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_text = f"uptime: {hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            uptime_text = "uptime: Not running"
        
        # Update CPU and Memory
        if stats['cpu'] is not None:
            cpu_text = f"CPU: {stats['cpu']:.1f}%"
            if stats['memory_percent'] is not None:
                cpu_text += f" | RAM: {stats['memory_used']:.1f}/{stats['memory_total']:.1f}GB ({stats['memory_percent']:.1f}%)"
        else:
            cpu_text = "CPU: --"
        
        # Update GPU info
        if stats.get('gpu_util') is not None:
            gpu_text = f"GPU: {stats['gpu_util']}% | VRAM: {stats['vram_used']:.0f}/{stats['vram_total']:.0f}MB"
            # Replace the memory label with GPU info since it's more relevant
            self.memory_label.setText(gpu_text)
        else:
            self.memory_label.setText("GPU: Not available")
        
        self.uptime_label.setText(uptime_text)
        self.cpu_label.setText(cpu_text)

        
    def update_status_indicators(self):
        """Update status indicators based on last health check."""
        # Update Redis status
        redis_healthy = self.last_health_status.get('redis', False)
        if not self.services_running:
            self.redis_status.update_status("stopped")
        else:
            self.redis_status.update_status("running" if redis_healthy else "error")
        
        # Update service status widgets
        for name, widget in self.service_widgets.items():
            if not self.services_running:
                widget.update_status("stopped")
            else:
                # Check if process is running
                process = self.processes.get(name)
                if process is None or process.poll() is not None:
                    widget.update_status("error")  # Process died
                else:
                    # Process is running, check health
                    service_healthy = self.last_health_status.get(name, False)
                    
                    # Special handling for Gary - if port is listening but health fails,
                    # it's probably just busy generating audio
                    if name == "gary" and not service_healthy:
                        port = self.services[name]["port"]
                        if self.check_port_listening(port):
                            widget.update_status("starting")  # Show as "busy" rather than "error"
                        else:
                            widget.update_status("error")
                    elif service_healthy:
                        widget.update_status("running")
                    else:
                        widget.update_status("starting")  # Process running but not healthy yet
                
        # Update main buttons based on overall system status
        # Enable start button only if not running and not currently starting
        self.start_btn.setEnabled(not self.services_running and not self.service_worker.isRunning())
        
        # Enable stop button if services are running or starting
        self.stop_btn.setEnabled(self.services_running and not self.service_worker.isRunning())
        
        # Update button text to show current state
        if self.service_worker.isRunning():
            if self.services_running:
                self.stop_btn.setText("stopping...")
            else:
                self.start_btn.setText("starting...")
        else:
            self.start_btn.setText("start")
            self.stop_btn.setText("stop")
        
    def apply_styling(self):
        """Apply modern dark styling to match JUCE plugin."""
        style = """
        QMainWindow {
            background-color: #000000;
            color: #FFFFFF;
        }
        
        QWidget {
            background-color: #000000;
            color: #FFFFFF;
        }
        
        QGroupBox {
            font-weight: bold;
            font-size: 12px;
            color: #FFFFFF;
            border: 2px solid #444444;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 15px;
            background-color: #2A2A2A;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 8px 0 8px;
            color: #FFFFFF;
            background-color: #2A2A2A;
        }
        
        QPushButton {
            background-color: #BB0000;
            color: #FFFFFF;
            border: none;
            padding: 10px 20px;
            border-radius: 0px;
            font-weight: bold;
            font-size: 11px;
        }
        
        QPushButton:hover {
            background-color: #FF6666;
        }
        
        QPushButton:pressed {
            background-color: #CC3333;
        }
        
        QPushButton:disabled {
            background-color: #555555;
            color: #999999;
        }
        
        QFrame {
            background-color: #2A2A2A;
            border-radius: 0px;
            border: 1px solid #444444;
        }
        
        QLabel {
            color: #FFFFFF;
        }
        
        QSplitter::handle {
            background-color: #444444;
            width: 3px;
        }
        
        QSplitter::handle:hover {
            background-color: #FF4444;
        }
        
        QScrollBar:vertical {
            background-color: #2A2A2A;
            width: 12px;
            border-radius: 0px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #555555;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #777777;
        }
        """
        
        self.setStyleSheet(style)
        
    # Health checking methods (unchanged)
    def check_redis_health(self):
        """Enhanced Redis health check with fallback strategies."""
        # First try with redis-py if available
        try:
            import redis
            # Try multiple potential Redis configurations
            redis_configs = [
                {"host": "localhost", "port": 6379, "db": 0},
                {"host": "127.0.0.1", "port": 6379, "db": 0},
                {"host": "localhost", "port": 6380, "db": 0},  # Alternative port
            ]
            
            for config in redis_configs:
                try:
                    r = redis.Redis(**config, decode_responses=True, socket_timeout=2)
                    response = r.ping()
                    if response:
                        return True
                except (redis.ConnectionError, redis.TimeoutError):
                    continue
                except Exception:
                    continue
                    
        except ImportError:
            pass
        
        # Fallback to port checking if redis-py not available
        ports_to_try = [6379, 6380]
        for port in ports_to_try:
            if self.check_port_listening(port):
                # Double-check it's actually Redis by trying a simple TCP connection
                try:
                    import socket
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(2)
                    sock.connect(('localhost', port))
                    
                    # Send a simple Redis command to verify it's actually Redis
                    sock.send(b"*1\r\n$4\r\nPING\r\n")
                    response = sock.recv(1024)
                    sock.close()
                    
                    # Check for Redis PONG response
                    if b"+PONG" in response:
                        return True
                        
                except Exception:
                    continue
        
        return False
        
    def check_port_listening(self, port):
        """Check if a port is listening (simple socket test)."""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
        
    def check_service_health(self, port, timeout=2):
        """Check if a service is responding via health endpoint."""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.ConnectionError as e:
            # Service not accepting connections yet - this is expected during startup
            return False
        except requests.exceptions.Timeout:
            # Service too slow to respond
            return False
        except Exception as e:
            # Other errors (like service returning 500)
            return False
            
    def check_service_health_detailed(self, port, timeout=2):
        """Detailed health check for debugging."""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
            return {
                "success": True,
                "status_code": response.status_code,
                "healthy": response.status_code == 200,
                "response": response.text[:200] if response.text else "Empty response"
            }
        except requests.exceptions.ConnectionError as e:
            return {
                "success": False,
                "error": f"Connection refused: {str(e)[:100]}",
                "healthy": False
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Timeout",
                "healthy": False
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Other error: {str(e)[:100]}",
                "healthy": False
            }
            
    def get_system_status(self):
        """Get overall system status using same intelligent logic as control panel."""
        if not self.services_running:
            return "stopped"
            
        # Check if service processes are still running (exclude Redis from this check)
        dead_processes = 0
        for name, process in self.processes.items():
            if name != "redis" and process and process.poll() is not None:
                dead_processes += 1
        
        if dead_processes > 0:
            return "error"
        
        # Use same intelligent logic as control panel - fall back to port checking
        healthy_count = 0
        total_services = len(self.services)
        
        for name in self.services.keys():
            process = self.processes.get(name)
            if process and process.poll() is None:  # Process is running
                # Check health status first
                service_healthy = self.last_health_status.get(name, False)
                
                if service_healthy:
                    healthy_count += 1
                else:
                    # Health check failed, fall back to port checking
                    port = self.services[name]["port"]
                    if self.check_port_listening(port):
                        healthy_count += 1  # Consider healthy if port is listening
        
        # Also check Redis
        if self.last_health_status.get('redis', False):
            redis_healthy = True
        else:
            redis_healthy = self.check_port_listening(6379)
                
        if healthy_count == total_services and redis_healthy:
            return "running"
        elif healthy_count > 0 or redis_healthy:
            return "starting"
        else:
            return "error"
            
    def update_status(self):
        """Update all status indicators (non-blocking)."""
        self.update_status_indicators()
        
    def add_log(self, service_name, message):
        """Add a log message to the viewer."""
        # Add timestamp for system messages
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.log_viewer.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def clear_logs(self):
        """Clear the log viewer."""
        self.log_viewer.clear()
        
    def start_services(self):
        """start all services using worker thread."""
        if self.services_running or self.service_worker.isRunning():
            return
            
        # Disable start button immediately
        self.start_btn.setEnabled(False)
        self.add_log("system", "⏳ starting services...")
        
        # Start services in background thread
        self.service_worker.start_services()
        
    def stop_services(self):
        """Stop all services using worker thread."""
        if not self.services_running or self.service_worker.isRunning():
            return
            
        # Disable stop button immediately
        self.stop_btn.setEnabled(False)
        self.add_log("system", "⏳ stopping services...")
        
        # Stop services in background thread
        self.service_worker.stop_services()
        
    def stop_services_sync(self):
        """Stop all services synchronously (for shutdown)."""
        if not self.services_running:
            return True
            
        self.add_log("system", "⏳ stopping services for shutdown...")
        
        # If worker is already running, wait for it to finish first
        if self.service_worker.isRunning():
            self.add_log("system", "⏳ waiting for current operation to complete...")
            self.service_worker.wait(10000)  # Wait up to 10 seconds
        
        # Stop processes with proper timeout handling
        self.add_log("system", "⏳ terminating service processes...")
        for name, process in self.processes.items():
            if process and process.poll() is None:
                try:
                    config = self.services.get(name, {})
                    service_name = config.get("name", name)
                    self.add_log("system", f"🛑 stopping {service_name}...")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                        self.add_log("system", f"✅ stopped {service_name}")
                    except subprocess.TimeoutExpired:
                        self.add_log("system", f"⚠️ force killing {service_name}...")
                        process.kill()
                        process.wait(timeout=2)
                        self.add_log("system", f"🛑 force stopped {service_name}")
                except Exception as e:
                    self.add_log("system", f"❌ error stopping {name}: {e}")
        
        # Force kill any remaining Python processes from our services
        self.add_log("system", "⏳ Cleaning up any remaining processes...")
        try:
            # Kill redis-server processes
            subprocess.run(["taskkill", "/F", "/IM", "redis-server.exe"], 
                         capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            
            # Kill any Python processes from our service directories
            for name, config in self.services.items():
                python_exe = config["dir"] / "env" / "Scripts" / "python.exe"
                if python_exe.exists():
                    # Get the exact python.exe path and kill processes using it
                    cmd = f'taskkill /F /FI "IMAGENAME eq python.exe" /FI "COMMANDLINE eq *{python_exe}*"'
                    subprocess.run(cmd, shell=True, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as e:
            self.add_log("system", f"⚠️ Cleanup warning: {e}")
                    
        self.processes.clear()
        self.services_running = False
        self.add_log("system", "✅ All services stopped!")
        return True
    
    def stop_service(self, service_name):
        """Stop a specific service without restarting."""
        if service_name not in self.services:
            return
            
        config = self.services[service_name]
        self.add_log("system", f"⏹️ stopping {config['name']}...")
        
        # Stop the specific service
        if service_name in self.processes:
            process = self.processes[service_name]
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    self.add_log("system", f"🛑 stopped {config['name']}")
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                    self.add_log("system", f"🛑 force stopped {config['name']}")
                except Exception as e:
                    self.add_log("system", f"❌ error stopping {config['name']}: {e}")
            
            # Remove from processes dict
            del self.processes[service_name]
            self.add_log("system", f"✅ {config['name']} stopped successfully")
        else:
            self.add_log("system", f"⚠️ {config['name']} is not running")
        
    def restart_service(self, service_name):
        """Restart a specific service."""
        if service_name not in self.services:
            return
            
        config = self.services[service_name]
        self.add_log("system", f"restarting {config['name']}...")
        
        # Stop the specific service
        if service_name in self.processes:
            process = self.processes[service_name]
            if process and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    self.add_log("system", f"🛑 Stopped {config['name']}")
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                    self.add_log("system", f"🛑 Force stopped {config['name']}")
                except Exception as e:
                    self.add_log("system", f"❌ Error stopping {config['name']}: {e}")
        
        # Start the service again
        try:
            python_exe = config["dir"] / "env" / "Scripts" / "python.exe"
            script_path = config["dir"] / config["script"]
            
            if not python_exe.exists() or not script_path.exists():
                self.add_log("system", f"❌ {config['name']} files not found")
                return
            
            self.add_log("system", f"▶️ Starting {config['name']} on port {config['port']}...")
            
            # Load environment variables
            env_vars = os.environ.copy()
            env_file = self.base_dir / ".env"
            if env_file.exists():
                env_vars.update(dotenv_values(str(env_file)))
            
            # Special environment for MelodyFlow
            if service_name == "melodyflow":
                env_vars["XFORMERS_DISABLED"] = "1"
                env_vars["FLASH_ATTENTION_DISABLED"] = "1"
            
            # Fix Unicode encoding issues on Windows
            env_vars["PYTHONIOENCODING"] = "utf-8"
            env_vars["PYTHONLEGACYWINDOWSSTDIO"] = "1"
            
            process = subprocess.Popen(
                [str(python_exe), str(script_path)],
                cwd=str(config["dir"]),
                env=env_vars,
                stdout=subprocess.DEVNULL,  # Ignore output
                stderr=subprocess.DEVNULL,  # Ignore errors
                creationflags=subprocess.CREATE_NO_WINDOW,
                text=True
            )
            
            self.processes[service_name] = process
            self.add_log("system", f"✅ Restarted {config['name']}")
            
        except Exception as e:
            self.add_log("system", f"❌ Error restarting {config['name']}: {e}")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.tray_icon.isVisible():
            # Hide to tray instead of closing, but ask user first if services are running
            if self.services_running:
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle('gary4juce control center')
                msg_box.setText('backend services are still running.')
                msg_box.setInformativeText('minimize or shutdown?')
                
                # Create custom buttons with clear action labels
                minimize_btn = msg_box.addButton('minimize to tray', QMessageBox.ButtonRole.AcceptRole)
                exit_btn = msg_box.addButton('shutdown and exit', QMessageBox.ButtonRole.RejectRole)
                
                # Set minimize as default (safer option)
                msg_box.setDefaultButton(minimize_btn)
                
                msg_box.exec()
                
                if msg_box.clickedButton() == exit_btn:
                    self.quit_application()
                    return
            
            # Hide to tray
            self.hide()
            event.ignore()
        else:
            self.quit_application()
            
    def quit_application(self):
        """Clean shutdown of the application."""
        self.add_log("system", "🔄 initiating shutdown sequence...")
        
        # Stop services synchronously first
        if self.services_running:
            self.stop_services_sync()
        
        # Stop health checker
        self.add_log("system", "⏳ stopping health checker...")
        if hasattr(self, 'health_checker'):
            self.health_checker.stop()
            self.health_checker.wait(3000)  # Wait up to 3 seconds
            
        # Stop service worker if still running
        self.add_log("system", "⏳ stopping service worker...")
        if hasattr(self, 'service_worker') and self.service_worker.isRunning():
            self.service_worker.terminate()
            self.service_worker.wait(3000)
            
        self.add_log("system", "✅ shutdown complete!")
        
        # Small delay to let the log appear
        QApplication.processEvents()
        time.sleep(0.5)
            
        QApplication.quit()

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties FIRST
    app.setApplicationName("gary4juce control center")
    app.setApplicationVersion("1.0")
    app.setQuitOnLastWindowClosed(False)
    
    # Set application icon BEFORE creating any windows - this affects taskbar
    base_dir = Path(__file__).parent
    icon_files = [
        base_dir / "icon.ico",
        base_dir / "icon.png",
    ]
    
    app_icon = None
    for icon_path in icon_files:
        if icon_path.exists():
            try:
                app_icon = QIcon(str(icon_path))
                # This sets the icon for ALL windows in the application
                app.setWindowIcon(app_icon)
                print(f"✅ Set application icon: {icon_path.name}")
                break
            except Exception as e:
                print(f"❌ Failed to load {icon_path.name}: {e}")
                continue
    
    if app_icon is None:
        print("⚠️ No icon files found")
    
    # Check if system tray is available
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, "System Tray", 
                           "System tray is not available on this system.")
        sys.exit(1)
    
    # Create and show control center
    control_center = Gary4JUCEControlCenter()
    control_center.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()