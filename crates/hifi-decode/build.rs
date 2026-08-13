// Locate libsoxr and emit the link flags for `crate::soxr`'s FFI binding.
//
// Bit-exact parity with `vhsdecode.hifi` requires resampling through the
// same C library Python's `soxr` package wraps, so this is a hard
// dependency rather than an optional acceleration — see `src/soxr.rs`.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if let Err(e) = pkg_config::Config::new().atleast_version("0.1.3").probe("soxr") {
        panic!(
            "libsoxr not found via pkg-config: {e}\n\
             hifi-decode resamples through libsoxr to stay byte-identical to \
             vhsdecode.hifi's output. Install it first:\n\
             \x20 macOS:         brew install libsoxr\n\
             \x20 Debian/Ubuntu: apt install libsoxr-dev\n\
             \x20 Fedora:        dnf install soxr-devel\n\
             \x20 Arch:          pacman -S libsoxr"
        );
    }
}
