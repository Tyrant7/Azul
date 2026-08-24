//! Child-process ownership and stream lifecycle for engine sessions.

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    process::{Child, ChildStdin, Command, ExitStatus, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Owns one engine process and its protocol and diagnostic streams.
pub(crate) struct EngineProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: Receiver<io::Result<String>>,
    stderr: Receiver<io::Result<String>>,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
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
        })
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
        stdin.flush()
    }

    /// Reads one protocol line from stdout, waiting up to `timeout`.
    pub(crate) fn recv_stdout(&mut self, timeout: Duration) -> io::Result<Option<String>> {
        receive_line(&self.stdout, timeout)
    }

    /// Reads one diagnostic line from stderr, waiting up to `timeout`.
    pub(crate) fn recv_stderr(&mut self, timeout: Duration) -> io::Result<Option<String>> {
        receive_line(&self.stderr, timeout)
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
    use super::EngineProcess;
    use std::{io::ErrorKind, process::Command, time::Duration};

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
        Duration::from_millis(50)
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
}
