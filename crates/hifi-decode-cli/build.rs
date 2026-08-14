// Locate libsndfile and emit the link flags for `crate::sndfile`'s FFI
// binding.
//
// Output goes through the same library `vhsdecode.hifi` writes with, so
// the encoder settings, sample conversion and container metadata match
// without being re-derived — see `src/sndfile.rs`.
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if let Err(e) = pkg_config::Config::new().atleast_version("1.0.28").probe("sndfile") {
        panic!(
            "libsndfile not found via pkg-config: {e}\n\
             hifi-decode writes its output through libsndfile to stay \
             byte-identical to vhsdecode.hifi's. Install it first:\n\
             \x20 macOS:         brew install libsndfile\n\
             \x20 Debian/Ubuntu: apt install libsndfile1-dev\n\
             \x20 Fedora:        dnf install libsndfile-devel\n\
             \x20 Arch:          pacman -S libsndfile"
        );
    }
}
