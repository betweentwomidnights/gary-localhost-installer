//! Windows process-lifetime boundary for GPU and other managed workloads.
//!
//! Managed roots are created suspended, assigned to a kill-on-close Job Object,
//! and then resumed. Their descendants inherit the job automatically, so an
//! abnormal control-center exit cannot leave Python workers behind.

#[cfg(target_os = "windows")]
mod platform {
    use std::io;
    use std::mem::{size_of, zeroed};
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::sync::OnceLock;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

    const CREATE_SUSPENDED: u32 = 0x0000_0004;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    struct OwnedHandle(HANDLE);

    unsafe impl Send for OwnedHandle {}
    unsafe impl Sync for OwnedHandle {}

    impl Drop for OwnedHandle {
        fn drop(&mut self) {
            if !self.0.is_null() && self.0 != INVALID_HANDLE_VALUE {
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }

    struct WorkloadJob {
        handle: OwnedHandle,
    }

    impl WorkloadJob {
        fn create() -> Result<Self, String> {
            let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
            if handle.is_null() {
                return Err(format!(
                    "Could not create workload Job Object: {}",
                    io::Error::last_os_error()
                ));
            }
            let handle = OwnedHandle(handle);
            let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { zeroed() };
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = unsafe {
                SetInformationJobObject(
                    handle.0,
                    JobObjectExtendedLimitInformation,
                    &limits as *const _ as *const core::ffi::c_void,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            };
            if configured == 0 {
                return Err(format!(
                    "Could not configure workload Job Object: {}",
                    io::Error::last_os_error()
                ));
            }
            Ok(Self { handle })
        }

        fn assign(&self, process: HANDLE) -> Result<(), String> {
            if unsafe { AssignProcessToJobObject(self.handle.0, process) } == 0 {
                return Err(format!(
                    "Could not assign process to workload Job Object: {}",
                    io::Error::last_os_error()
                ));
            }
            Ok(())
        }
    }

    static WORKLOAD_JOB: OnceLock<Result<WorkloadJob, String>> = OnceLock::new();

    fn workload_job() -> Result<&'static WorkloadJob, String> {
        match WORKLOAD_JOB.get_or_init(WorkloadJob::create) {
            Ok(job) => Ok(job),
            Err(error) => Err(error.clone()),
        }
    }

    fn resume_process(pid: u32) -> Result<(), String> {
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(format!(
                "Could not enumerate threads for managed process {pid}: {}",
                io::Error::last_os_error()
            ));
        }
        let snapshot = OwnedHandle(snapshot);
        let mut entry: THREADENTRY32 = unsafe { zeroed() };
        entry.dwSize = size_of::<THREADENTRY32>() as u32;
        let mut has_entry = unsafe { Thread32First(snapshot.0, &mut entry) } != 0;
        let mut resumed = 0usize;

        while has_entry {
            if entry.th32OwnerProcessID == pid {
                let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) };
                if thread.is_null() {
                    return Err(format!(
                        "Could not open suspended thread {} for managed process {pid}: {}",
                        entry.th32ThreadID,
                        io::Error::last_os_error()
                    ));
                }
                let thread = OwnedHandle(thread);
                if unsafe { ResumeThread(thread.0) } == u32::MAX {
                    return Err(format!(
                        "Could not resume thread {} for managed process {pid}: {}",
                        entry.th32ThreadID,
                        io::Error::last_os_error()
                    ));
                }
                resumed += 1;
            }
            has_entry = unsafe { Thread32Next(snapshot.0, &mut entry) } != 0;
        }

        if resumed == 0 {
            return Err(format!(
                "Managed process {pid} was created suspended but no thread could be resumed"
            ));
        }
        Ok(())
    }

    fn enroll_and_resume(process: HANDLE, pid: u32) -> Result<(), String> {
        workload_job()?.assign(process)?;
        resume_process(pid)
    }

    pub fn configure_std_command(command: &mut std::process::Command) {
        command.creation_flags(CREATE_NO_WINDOW | CREATE_SUSPENDED);
    }

    pub fn configure_tokio_command(command: &mut tokio::process::Command) {
        command.creation_flags(CREATE_NO_WINDOW | CREATE_SUSPENDED);
    }

    pub fn enroll_std_child(child: &std::process::Child) -> Result<(), String> {
        enroll_and_resume(child.as_raw_handle() as HANDLE, child.id())
    }

    pub fn enroll_tokio_child(child: &tokio::process::Child) -> Result<(), String> {
        let pid = child
            .id()
            .ok_or_else(|| "Managed process has no process ID".to_string())?;
        let process = child
            .raw_handle()
            .ok_or_else(|| format!("Managed process {pid} has no process handle"))?;
        enroll_and_resume(process as HANDLE, pid)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::time::{Duration, Instant};

        #[test]
        fn closing_job_terminates_a_managed_process() {
            let mut command = std::process::Command::new("powershell");
            command.args(["-NoProfile", "-Command", "Start-Sleep -Seconds 30"]);
            configure_std_command(&mut command);
            let mut child = command.spawn().expect("spawn suspended test process");
            let pid = child.id();
            let job = WorkloadJob::create().expect("create test workload job");
            job.assign(child.as_raw_handle() as HANDLE)
                .expect("assign test process");
            resume_process(pid).expect("resume test process");

            drop(job);
            let deadline = Instant::now() + Duration::from_secs(3);
            loop {
                if child.try_wait().expect("query test process").is_some() {
                    break;
                }
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("managed process survived after its Job Object closed");
                }
                std::thread::sleep(Duration::from_millis(25));
            }
        }

        #[test]
        fn shared_managed_spawn_enrolls_and_resumes() {
            let mut command = std::process::Command::new("powershell");
            command.args(["-NoProfile", "-Command", "exit 0"]);
            configure_std_command(&mut command);
            let mut child = command.spawn().expect("spawn suspended managed process");
            enroll_std_child(&child).expect("enroll and resume managed process");
            assert!(child.wait().expect("wait for managed process").success());
        }
    }
}

#[cfg(not(target_os = "windows"))]
mod platform {
    pub fn configure_std_command(_command: &mut std::process::Command) {}
    pub fn configure_tokio_command(_command: &mut tokio::process::Command) {}
    pub fn enroll_std_child(_child: &std::process::Child) -> Result<(), String> {
        Ok(())
    }
    pub fn enroll_tokio_child(_child: &tokio::process::Child) -> Result<(), String> {
        Ok(())
    }
}

pub use platform::{
    configure_std_command, configure_tokio_command, enroll_std_child, enroll_tokio_child,
};
