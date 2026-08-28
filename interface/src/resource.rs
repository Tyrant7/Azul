//! Platform-specific process resource limits for engine children.

use std::{
    io,
    process::{Child, Command},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

/// Resource limits applied to one engine process and its descendants where supported.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ProcessLimits {
    /// Maximum process memory in mebibytes.
    pub(crate) memory_mib: Option<u64>,
    /// Maximum number of live engine threads.
    pub(crate) threads: Option<u32>,
}

impl ProcessLimits {
    /// Returns limits with no enforcement, used by low-level process tests.
    pub(crate) const fn unrestricted() -> Self {
        Self {
            memory_mib: None,
            threads: None,
        }
    }

    /// Validates and converts a CLI resource-limit configuration.
    pub(crate) fn from_config(memory_mib: Option<u64>, threads: u32) -> io::Result<Self> {
        if memory_mib == Some(0) || threads == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "memory and thread limits must be greater than zero",
            ));
        }
        Ok(Self {
            memory_mib,
            threads: Some(threads),
        })
    }

    /// Installs pre-spawn limits that must be configured before process creation.
    pub(crate) fn configure_command(&self, command: &mut Command) -> io::Result<()> {
        #[cfg(not(target_os = "linux"))]
        let _ = command;
        #[cfg(target_os = "linux")]
        if let Some(memory_mib) = self.memory_mib {
            use std::os::unix::process::CommandExt;

            let bytes = memory_mib.checked_mul(1024 * 1024).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "memory limit is too large")
            })?;
            unsafe {
                command.pre_exec(move || {
                    let limit = libc::rlimit {
                        rlim_cur: bytes,
                        rlim_max: bytes,
                    };
                    if libc::setrlimit(libc::RLIMIT_AS, &limit) == 0 {
                        Ok(())
                    } else {
                        Err(io::Error::last_os_error())
                    }
                });
            }
        }

        #[cfg(all(not(target_os = "linux"), not(windows)))]
        if self.memory_mib.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "memory limits are not implemented on this platform",
            ));
        }

        Ok(())
    }

    /// Attaches post-spawn limits and starts enforcement monitors.
    pub(crate) fn attach(&self, child: &Child) -> io::Result<ResourceGuard> {
        #[cfg(windows)]
        {
            windows_limits::attach(self, child)
        }

        #[cfg(target_os = "linux")]
        {
            linux_limits::attach(self, child)
        }

        #[cfg(all(not(target_os = "linux"), not(windows)))]
        {
            if self.threads.is_some() {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "thread limits are not implemented on this platform",
                ));
            }
            Ok(ResourceGuard::none_for_process())
        }
    }
}

/// Owns platform resource handles and stops their monitor threads.
pub(crate) struct ResourceGuard {
    stop: Arc<AtomicBool>,
    violated: Arc<AtomicBool>,
    monitor: Option<JoinHandle<()>>,
    #[cfg(windows)]
    job: windows_limits::JobHandle,
}

impl ResourceGuard {
    /// Creates an inactive guard for an unrestricted process.
    pub(crate) fn none_for_process() -> Self {
        Self {
            stop: Arc::new(AtomicBool::new(true)),
            violated: Arc::new(AtomicBool::new(false)),
            monitor: None,
            #[cfg(windows)]
            job: windows_limits::JobHandle::none(),
        }
    }

    /// Returns true when the monitor terminated the process for a limit violation.
    pub(crate) fn violated(&self) -> bool {
        self.violated.load(Ordering::Acquire)
    }
}

impl Drop for ResourceGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(monitor) = self.monitor.take() {
            let _ = monitor.join();
        }
    }
}

#[cfg(target_os = "linux")]
mod linux_limits {
    use super::*;

    /// Attaches a Linux thread-count monitor to the child process.
    pub(super) fn attach(limits: &ProcessLimits, child: &Child) -> io::Result<ResourceGuard> {
        let stop = Arc::new(AtomicBool::new(false));
        let violated = Arc::new(AtomicBool::new(false));
        let monitor = limits.threads.map(|thread_limit| {
            let stop = Arc::clone(&stop);
            let violated = Arc::clone(&violated);
            let pid = child.id();
            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    if thread_count(pid).is_some_and(|count| count > thread_limit) {
                        violated.store(true, Ordering::Release);
                        unsafe {
                            libc::kill(pid as i32, libc::SIGKILL);
                        }
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            })
        });
        Ok(ResourceGuard {
            stop,
            violated,
            monitor,
        })
    }

    /// Reads the live thread count from Linux's process status file.
    fn thread_count(pid: u32) -> Option<u32> {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        status
            .lines()
            .find_map(|line| line.strip_prefix("Threads:")?.trim().parse().ok())
    }
}

#[cfg(windows)]
mod windows_limits {
    use super::*;
    use std::{mem, os::windows::io::AsRawHandle, ptr};
    use windows_sys::Win32::{
        Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE},
        System::{
            Diagnostics::ToolHelp::{
                CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First,
                Thread32Next,
            },
            JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                JobObjectExtendedLimitInformation, SetInformationJobObject,
            },
            Threading::{OpenProcess, PROCESS_TERMINATE, TerminateProcess},
        },
    };

    /// Owns the Windows Job Object used for memory enforcement.
    pub(super) struct JobHandle(HANDLE);

    impl JobHandle {
        /// Creates an empty handle for an unrestricted process.
        pub(super) const fn none() -> Self {
            Self(ptr::null_mut())
        }
    }

    impl Drop for JobHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    CloseHandle(self.0);
                }
            }
        }
    }

    /// Creates the Job Object and thread monitor for a Windows child.
    pub(super) fn attach(limits: &ProcessLimits, child: &Child) -> io::Result<ResourceGuard> {
        let job = if let Some(memory_mib) = limits.memory_mib {
            let bytes = memory_mib.checked_mul(1024 * 1024).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "memory limit is too large")
            })?;
            unsafe {
                let handle = CreateJobObjectW(ptr::null(), ptr::null());
                if handle.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
                info.BasicLimitInformation.LimitFlags =
                    JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                info.ProcessMemoryLimit = usize::try_from(bytes).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidInput, "memory limit is too large")
                })?;
                let configured = SetInformationJobObject(
                    handle,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) != 0;
                let assigned =
                    configured && AssignProcessToJobObject(handle, child.as_raw_handle()) != 0;
                if !assigned {
                    CloseHandle(handle);
                    return Err(io::Error::last_os_error());
                }
                JobHandle(handle)
            }
        } else {
            JobHandle::none()
        };

        let stop = Arc::new(AtomicBool::new(false));
        let violated = Arc::new(AtomicBool::new(false));
        let monitor = limits.threads.map(|thread_limit| {
            let stop = Arc::clone(&stop);
            let violated = Arc::clone(&violated);
            let pid = child.id();
            thread::spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    if thread_count(pid).is_some_and(|count| count > thread_limit) {
                        violated.store(true, Ordering::Release);
                        unsafe {
                            let process = OpenProcess(PROCESS_TERMINATE, 0, pid);
                            if !process.is_null() {
                                TerminateProcess(process, 0xE001);
                                CloseHandle(process);
                            }
                        }
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            })
        });

        Ok(ResourceGuard {
            stop,
            violated,
            monitor,
            job,
        })
    }

    /// Counts threads owned by the child through the Toolhelp snapshot API.
    fn thread_count(pid: u32) -> Option<u32> {
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return None;
            }
            let mut entry = THREADENTRY32 {
                dwSize: mem::size_of::<THREADENTRY32>() as u32,
                ..Default::default()
            };
            let mut count = 0;
            if Thread32First(snapshot, &mut entry) != 0 {
                loop {
                    if entry.th32OwnerProcessID == pid {
                        count += 1;
                    }
                    if Thread32Next(snapshot, &mut entry) == 0 {
                        break;
                    }
                }
            }
            CloseHandle(snapshot);
            Some(count)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProcessLimits;

    #[test]
    fn accepts_positive_memory_and_thread_limits() {
        let limits = ProcessLimits::from_config(Some(512), 4).unwrap();

        assert_eq!(limits.memory_mib, Some(512));
        assert_eq!(limits.threads, Some(4));
    }

    #[test]
    fn rejects_zero_resource_limits() {
        assert!(ProcessLimits::from_config(Some(0), 1).is_err());
        assert!(ProcessLimits::from_config(None, 0).is_err());
    }
}
