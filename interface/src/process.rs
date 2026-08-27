//! Child-process ownership and stream lifecycle for engine sessions.

use std::{
    ffi::OsString,
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::PathBuf,
    process::{Child, ChildStdin, Command, ExitStatus, Stdio},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, RecvTimeoutError, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::resource::{ProcessLimits, ResourceGuard};

/// Owns one engine process and its protocol and diagnostic streams.
pub(crate) struct EngineProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<io::Result<String>>,
    stderr: Receiver<io::Result<String>>,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    diagnostics: Option<Arc<ProcessDiagnostics>>,
    engine_index: usize,
    resources: ResourceGuard,
}

/// Shares engine stream logging and optional stderr display across processes.
pub(crate) struct ProcessDiagnostics {
    show_stderr: bool,
    show_protocol: bool,
    log: Option<Mutex<BufWriter<File>>>,
}

impl ProcessDiagnostics {
    /// Creates a diagnostics sink, truncating an existing log file.
    pub(crate) fn new(
        show_stderr: bool,
        show_protocol: bool,
        log_path: Option<PathBuf>,
    ) -> io::Result<Self> {
        let log = log_path
            .map(File::create)
            .transpose()?
            .map(BufWriter::new)
            .map(Mutex::new);
        Ok(Self {
            show_stderr,
            show_protocol,
            log,
        })
    }

    /// Records one command or stream line for an engine.
    fn record(&self, engine_index: usize, direction: &str, line: &str) -> io::Result<()> {
        if direction == "stderr" && self.show_stderr {
            eprintln!("[engine {engine_index} stderr] {line}");
        } else if direction != "stderr" && self.show_protocol {
            eprintln!("[engine {engine_index} {direction}] {line}");
        }
        if let Some(log) = &self.log {
            let mut log = log
                .lock()
                .map_err(|_| io::Error::other("engine log lock was poisoned"))?;
            writeln!(log, "[engine {engine_index} {direction}] {line}")?;
            log.flush()?;
        }
        Ok(())
    }
}

/// Describes how to launch one engine process.
#[derive(Debug, Clone)]
pub(crate) struct EngineLaunch {
    program: OsString,
    args: Vec<OsString>,
    current_dir: Option<PathBuf>,
    limits: ProcessLimits,
}

impl EngineLaunch {
    /// Creates a launch specification from CLI-compatible string values.
    pub(crate) fn new(program: String, args: Vec<String>, current_dir: Option<String>) -> Self {
        Self {
            program: program.into(),
            args: args.into_iter().map(OsString::from).collect(),
            current_dir: current_dir.map(PathBuf::from),
            limits: ProcessLimits::unrestricted(),
        }
    }

    /// Adds validated per-process memory and thread limits to this launch.
    pub(crate) fn with_limits(mut self, memory_mib: Option<u64>, threads: u32) -> io::Result<Self> {
        self.limits = ProcessLimits::from_config(memory_mib, threads)?;
        Ok(self)
    }

    /// Builds a command for a fresh engine process.
    fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        if let Some(current_dir) = &self.current_dir {
            command.current_dir(current_dir);
        }
        command
    }
}

impl EngineProcess {
    /// Starts an engine with piped standard input, output, and error streams.
    pub(crate) fn spawn(command: &mut Command) -> io::Result<Self> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "engine stdin was not piped")
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "engine stdout was not piped")
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            io::Error::new(io::ErrorKind::BrokenPipe, "engine stderr was not piped")
        })?;

        let (stdout_rx, stdout_thread) = spawn_reader(stdout);
        let (stderr_rx, stderr_thread) = spawn_reader(stderr);

        Ok(Self {
            child,
            stdin: Some(stdin),
            stdout: stdout_rx,
            stderr: stderr_rx,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
            diagnostics: None,
            engine_index: 0,
            resources: ResourceGuard::none_for_process(),
        })
    }

    /// Attaches shared diagnostics to this process.
    pub(crate) fn configure_diagnostics(
        &mut self,
        diagnostics: Option<Arc<ProcessDiagnostics>>,
        engine_index: usize,
    ) {
        self.diagnostics = diagnostics;
        self.engine_index = engine_index;
    }

    /// Starts an engine from a reusable launch specification.
    pub(crate) fn spawn_launch(launch: &EngineLaunch) -> io::Result<Self> {
        let mut command = launch.command();
        launch.limits.configure_command(&mut command)?;
        Self::spawn_with_limits(&mut command, launch.limits)
    }

    /// Starts a process and attaches the requested resource enforcement.
    fn spawn_with_limits(command: &mut Command, limits: ProcessLimits) -> io::Result<Self> {
        let mut process = Self::spawn(command)?;
        match limits.attach(&process.child) {
            Ok(resources) => {
                process.resources = resources;
                Ok(process)
            }
            Err(error) => {
                let _ = process.child.kill();
                let _ = process.child.wait();
                Err(error)
            }
        }
    }

    /// Replaces this process with a freshly spawned instance.
    ///
    /// The replacement is started before the old process is shut down so a
    /// failed spawn leaves the existing process available to the caller.
    pub(crate) fn restart(
        &mut self,
        launch: &EngineLaunch,
        shutdown_timeout: Duration,
    ) -> io::Result<()> {
        let replacement = Self::spawn_launch(launch)?;
        let mut replacement = replacement;
        replacement.configure_diagnostics(self.diagnostics.clone(), self.engine_index);
        let old = std::mem::replace(self, replacement);
        let _ = old.shutdown(shutdown_timeout);
        Ok(())
    }

    /// Writes one newline-terminated protocol command to the engine.
    pub(crate) fn send_line(&mut self, line: &str) -> io::Result<()> {
        if line.contains(['\r', '\n']) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "protocol commands cannot contain newlines",
            ));
        }
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "engine stdin is closed"))?;
        stdin.write_all(line.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        if let Some(diagnostics) = &self.diagnostics {
            diagnostics.record(self.engine_index, "command", line)?;
            self.drain_stderr()
        } else {
            Ok(())
        }
    }

    /// Reads one protocol line from stdout, waiting up to `timeout`.
    pub(crate) fn recv_stdout(&mut self, timeout: Duration) -> io::Result<Option<String>> {
        let result = receive_line(&self.stdout, timeout);
        if self.diagnostics.is_some() {
            self.drain_stderr()?;
        }
        if let Ok(Some(line)) = &result {
            if let Some(diagnostics) = &self.diagnostics {
                diagnostics.record(self.engine_index, "stdout", line)?;
            }
        }
        match result? {
            Some(line) => Ok(Some(line)),
            None if self.resources.violated() => Err(io::Error::other(
                "engine exceeded its memory or thread limit",
            )),
            None => Ok(None),
        }
    }

    /// Reads one diagnostic line from stderr, waiting up to `timeout`.
    pub(crate) fn recv_stderr(&mut self, timeout: Duration) -> io::Result<Option<String>> {
        match receive_line(&self.stderr, timeout)? {
            Some(line) => {
                self.record_stderr(&line)?;
                Ok(Some(line))
            }
            None => Ok(None),
        }
    }

    /// Drains currently available diagnostic stderr lines without blocking.
    pub(crate) fn drain_stderr(&mut self) -> io::Result<()> {
        loop {
            match self.stderr.try_recv() {
                Ok(Ok(line)) => self.record_stderr(&line)?,
                Ok(Err(error)) => return Err(error),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return Ok(()),
            }
        }
    }

    /// Records one stderr line through the configured diagnostics sink.
    fn record_stderr(&self, line: &str) -> io::Result<()> {
        if let Some(diagnostics) = &self.diagnostics {
            diagnostics.record(self.engine_index, "stderr", line)?;
        }
        Ok(())
    }

    /// Returns the child exit status when it has exited without blocking.
    pub(crate) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child.try_wait()
    }

    /// Requests graceful shutdown and then waits up to `timeout` for exit.
    pub(crate) fn shutdown(mut self, timeout: Duration) -> io::Result<ExitStatus> {
        let _ = self.send_line("quit");
        let deadline = Instant::now() + timeout;

        loop {
            if let Some(status) = self.child.try_wait()? {
                return Ok(status);
            }
            if Instant::now() >= deadline {
                self.stdin.take();
                self.child.kill()?;
                return self.child.wait();
            }
            thread::sleep(
                Duration::from_millis(5).min(deadline.saturating_duration_since(Instant::now())),
            );
        }
    }
}

impl Drop for EngineProcess {
    fn drop(&mut self) {
        self.stdin.take();
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        if let Some(thread) = self.stdout_thread.take() {
            let _ = thread.join();
        }
        if let Some(thread) = self.stderr_thread.take() {
            let _ = thread.join();
        }
        let _ = self.drain_stderr();
    }
}

/// Reads newline-delimited output from one child-process stream.
fn spawn_reader<R>(reader: R) -> (Receiver<io::Result<String>>, JoinHandle<()>)
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    let thread = thread::spawn(move || {
        let mut reader = BufReader::new(reader);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    let line = line.trim_end_matches(['\r', '\n']).to_owned();
                    if sender.send(Ok(line)).is_err() {
                        break;
                    }
                }
                Err(error) => {
                    let _ = sender.send(Err(error));
                    break;
                }
            }
        }
    });
    (receiver, thread)
}

/// Receives one line from a child-process stream with a timeout.
fn receive_line(
    receiver: &Receiver<io::Result<String>>,
    timeout: Duration,
) -> io::Result<Option<String>> {
    match receiver.recv_timeout(timeout) {
        Ok(line) => line.map(Some),
        Err(RecvTimeoutError::Timeout) => Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "engine response timed out",
        )),
        Err(RecvTimeoutError::Disconnected) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::{EngineProcess, ProcessDiagnostics};
    use std::{
        fs,
        io::ErrorKind,
        process::Command,
        sync::Arc,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    /// Builds a platform-native child command for process lifecycle tests.
    fn fixture(script: &str) -> Command {
        #[cfg(windows)]
        {
            let mut command = Command::new(std::env::var_os("COMSPEC").unwrap());
            command.args(["/Q", "/V:ON", "/C", script]);
            command
        }

        #[cfg(not(windows))]
        {
            let mut command = Command::new("sh");
            command.args(["-c", script]);
            command
        }
    }

    /// Returns a short timeout used to distinguish a live child from EOF.
    fn short_timeout() -> Duration {
        Duration::from_millis(200)
    }

    #[test]
    fn round_trips_a_line_through_stdin_and_stdout() {
        #[cfg(windows)]
        let script = "set /p line=& echo stdout:!line!";
        #[cfg(not(windows))]
        let script = "IFS= read line; printf 'stdout:%s\\n' \"$line\"";

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        process.send_line("ping").unwrap();

        assert_eq!(
            process.recv_stdout(short_timeout()).unwrap(),
            Some(String::from("stdout:ping"))
        );
    }

    #[test]
    fn stdout_and_stderr_are_available_without_blocking_each_other() {
        #[cfg(windows)]
        let script = "echo stdout& 1>&2 echo stderr";
        #[cfg(not(windows))]
        let script = "printf 'stdout\\n'; printf 'stderr\\n' >&2";

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();

        assert_eq!(
            process.recv_stdout(short_timeout()).unwrap(),
            Some(String::from("stdout"))
        );
        assert_eq!(
            process.recv_stderr(short_timeout()).unwrap(),
            Some(String::from("stderr"))
        );
    }

    #[test]
    fn reports_stdout_eof_after_the_child_closes_its_output() {
        let mut process = EngineProcess::spawn(&mut fixture("exit 0")).unwrap();

        assert_eq!(process.recv_stdout(short_timeout()).unwrap(), None);
        let deadline = std::time::Instant::now() + short_timeout();
        while process.try_wait().unwrap().is_none() && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(process.try_wait().unwrap().is_some());
    }

    #[test]
    fn preserves_nonzero_exit_status() {
        #[cfg(windows)]
        let script = "exit /b 7";
        #[cfg(not(windows))]
        let script = "exit 7";

        let process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        let status = process.shutdown(short_timeout()).unwrap();

        assert_eq!(status.code(), Some(7));
    }

    #[test]
    fn reports_a_timeout_without_treating_it_as_eof() {
        #[cfg(windows)]
        let script = "ping -n 3 127.0.0.1 > nul";
        #[cfg(not(windows))]
        let script = "sleep 2";

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        let error = process.recv_stdout(short_timeout()).unwrap_err();

        assert_eq!(error.kind(), ErrorKind::TimedOut);
    }

    #[test]
    fn shutdown_sends_quit_and_waits_for_graceful_exit() {
        #[cfg(windows)]
        let script = "set /p line=& if \"!line!\"==\"quit\" exit /b 0";
        #[cfg(not(windows))]
        let script = "IFS= read line; [ \"$line\" = quit ]";

        let process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        let status = process.shutdown(short_timeout()).unwrap();

        assert!(status.success());
    }

    #[test]
    fn diagnostics_log_commands_and_child_streams() {
        #[cfg(windows)]
        let script = "set /p line=& echo stdout:!line!& 1>&2 echo stderr";
        #[cfg(not(windows))]
        let script = "IFS= read line; printf 'stdout:%s\\n' \"$line\"; printf 'stderr\\n' >&2";

        let log_path = std::env::temp_dir().join(format!(
            "azul-interface-log-{}.log",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let diagnostics =
            Arc::new(ProcessDiagnostics::new(false, false, Some(log_path.clone())).unwrap());
        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        process.configure_diagnostics(Some(diagnostics), 7);
        process.send_line("ping").unwrap();

        assert_eq!(
            process.recv_stdout(short_timeout()).unwrap(),
            Some(String::from("stdout:ping"))
        );
        drop(process);

        let log = fs::read_to_string(&log_path).unwrap();
        assert!(log.contains("[engine 7 command] ping"));
        assert!(log.contains("[engine 7 stdout] stdout:ping"));
        assert!(log.contains("[engine 7 stderr] stderr"));
        let _ = fs::remove_file(log_path);
    }
}
