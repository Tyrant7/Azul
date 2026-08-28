//! End-to-end tests for the interface executable.

use std::{
    env, fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

/// Returns the workspace random-engine executable built alongside the test.
fn random_engine_path() -> PathBuf {
    let test_binary = env::current_exe().expect("test executable path is available");
    let executable_name = if cfg!(windows) {
        "random_engine.exe"
    } else {
        "random_engine"
    };
    test_binary
        .parent()
        .and_then(|path| path.parent())
        .expect("test executable is inside target/debug/deps")
        .join(executable_name)
}

/// Creates a unique temporary results path for the interface invocation.
fn results_path() -> PathBuf {
    env::temp_dir().join(format!(
        "azul-interface-smoke-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the Unix epoch")
            .as_nanos()
    ))
}

#[test]
fn random_engines_complete_a_seeded_game_through_the_interface() {
    let random_engine = random_engine_path();
    assert!(
        random_engine.is_file(),
        "random engine was not built at {}",
        random_engine.display()
    );

    let results = results_path();
    let output = Command::new(env!("CARGO_BIN_EXE_interface"))
        .args([
            "--engine",
            &format!("path={} tc=60", random_engine.display()),
            &format!("path={} tc=60", random_engine.display()),
            "--out",
            results.to_str().expect("temporary path is valid UTF-8"),
            "--seed",
            "1",
            "--timeout",
            "5",
        ])
        .output()
        .expect("interface process starts");

    let _ = fs::remove_file(&results);
    assert!(
        output.status.success(),
        "interface failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("Game over"),
        "interface did not report a terminal game: {}",
        String::from_utf8_lossy(&output.stdout)
    );
}
