/// Retains the CUDA LibTorch dependency in Linux executables.
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").ok();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").ok();

    match (target_os.as_deref(), target_env.as_deref()) {
        (Some("linux"), _) => {
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed,-ltorch_cuda");
            println!("cargo:rustc-link-arg=-Wl,--undefined=_ZN2at4cuda9warp_sizeEv");
        }
        (Some("windows"), Some("msvc")) => {
            println!("cargo:rustc-link-arg=/INCLUDE:?warp_size@cuda@at@@YAHXZ");
        }
        _ => {}
    }
}
