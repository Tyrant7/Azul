//! Child-process ownership and stream lifecycle for engine sessions.

use std::{
    io,
    process::{Command, ExitStatus},
    time::Duration,
};

/// Owns one engine process and its protocol and diagnostic streams.
pub(crate) struct EngineProcess;

impl EngineProcess {
    /// Starts an engine with piped standard input, output, and error streams.
    pub(crate) fn spawn(_command: &mut Command) -> io::Result<Self> {
        todo!("child-process wiring is covered by the test contract below")
    }

    /// Writes one newline-terminated protocol command to the engine.
    pub(crate) fn send_line(&mut self, _line: &str) -> io::Result<()> {
        todo!("child-process stdin wiring is not implemented")
    }

    /// Reads one protocol line from stdout, waiting up to `timeout`.
    pub(crate) fn recv_stdout(&mut self, _timeout: Duration) -> io::Result<Option<String>> {
        todo!("child-process stdout wiring is not implemented")
    }

    /// Reads one diagnostic line from stderr, waiting up to `timeout`.
    pub(crate) fn recv_stderr(&mut self, _timeout: Duration) -> io::Result<Option<String>> {
        todo!("child-process stderr wiring is not implemented")
    }

    /// Returns the child exit status when it has exited without blocking.
    pub(crate) fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        todo!("child-process wait handling is not implemented")
    }

    /// Requests graceful shutdown and then waits up to `timeout` for exit.
    pub(crate) fn shutdown(self, _timeout: Duration) -> io::Result<ExitStatus> {
        todo!("child-process shutdown is not implemented")
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
            command.args(["/Q", "/C", script]);
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
        let script = "set /p line=& echo stdout:%line%";
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
        let script = "echo stdout& echo stderr 1>&2";
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
        assert!(process.try_wait().unwrap().is_some());
    }

    #[test]
    fn preserves_nonzero_exit_status() {
        #[cfg(windows)]
        let script = "exit /b 7";
        #[cfg(not(windows))]
        let script = "exit 7";

        let mut process = EngineProcess::spawn(&mut fixture(script)).unwrap();
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
        let script = "set /p line=& if \"%line%\"==\"quit\" exit /b 0";
        #[cfg(not(windows))]
        let script = "IFS= read line; [ \"$line\" = quit ]";

        let process = EngineProcess::spawn(&mut fixture(script)).unwrap();
        let status = process.shutdown(short_timeout()).unwrap();

        assert!(status.success());
    }
}
