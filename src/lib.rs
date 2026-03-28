#![doc = "Nuvai fork of bindgen_cuda — graceful CUDA kernel building for Rust."]

use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub mod compute_cap;
pub mod error;

use error::{Error, Result};

/// Core builder for setting up CUDA kernel compilation options.
#[derive(Debug)]
pub struct Builder {
    cuda_root: Option<PathBuf>,
    kernel_paths: Vec<PathBuf>,
    watch: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
    compute_cap: Option<usize>,
    out_dir: PathBuf,
    extra_args: Vec<String>,
}

impl Default for Builder {
    fn default() -> Self {
        let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
            |_| num_cpus::get_physical(),
            |s| usize::from_str(&s).expect("RAYON_NUM_THREADS is not set to a valid integer"),
        );

        // Tolerate rayon already being initialized (e.g. when building multiple targets)
        if rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus)
            .build_global()
            .is_err()
        {
            // Already initialized — that's fine
        }

        let out_dir: PathBuf = std::env::var("OUT_DIR")
            .expect(
                "Expected OUT_DIR environment variable to be present. \
                 Is this running within `build.rs`?",
            )
            .into();

        let cuda_root = cuda_include_dir();
        let kernel_paths = default_kernels().unwrap_or_default();
        let include_paths = default_include().unwrap_or_default();
        let extra_args = vec![];
        let watch = vec![];

        // Detect compute capability — None if detection fails (not a panic)
        let compute_cap = compute_cap::detect().ok().map(|c| c.as_flat());

        Self {
            cuda_root,
            kernel_paths,
            watch,
            include_paths,
            extra_args,
            compute_cap,
            out_dir,
        }
    }
}

/// Helper struct returned by [`Builder::build_ptx`]. Contains compiled PTX paths
/// and can write a Rust source file with `include_str!` constants.
pub struct Bindings {
    write: bool,
    paths: Vec<PathBuf>,
}

fn default_kernels() -> Option<Vec<PathBuf>> {
    Some(
        glob::glob("src/**/*.cu")
            .ok()?
            .map(|p| p.expect("Invalid path"))
            .collect(),
    )
}

fn default_include() -> Option<Vec<PathBuf>> {
    Some(
        glob::glob("src/**/*.cuh")
            .ok()?
            .map(|p| p.expect("Invalid path"))
            .collect(),
    )
}

impl Builder {
    /// Set kernel source paths. All paths must exist.
    pub fn kernel_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Self {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        let missing: Vec<_> = paths.iter().filter(|f| !f.exists()).collect();
        if !missing.is_empty() {
            panic!("Kernel paths do not exist: {missing:?}");
        }
        self.kernel_paths = paths;
        self
    }

    /// Set paths to watch for changes (triggers recompilation).
    pub fn watch<T, P>(mut self, paths: T) -> Self
    where
        T: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        let missing: Vec<_> = paths.iter().filter(|f| !f.exists()).collect();
        if !missing.is_empty() {
            panic!("Watch paths do not exist: {missing:?}");
        }
        self.watch = paths;
        self
    }

    /// Set include header paths.
    pub fn include_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Self {
        self.include_paths = paths.into_iter().map(|p| p.into()).collect();
        self
    }

    /// Set kernel paths via a glob pattern.
    pub fn kernel_paths_glob(mut self, glob: &str) -> Self {
        self.kernel_paths = glob::glob(glob)
            .expect("Invalid glob pattern")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    /// Set include paths via a glob pattern.
    pub fn include_paths_glob(mut self, glob: &str) -> Self {
        self.include_paths = glob::glob(glob)
            .expect("Invalid glob pattern")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    /// Override the output directory (defaults to `OUT_DIR`).
    pub fn out_dir<P: Into<PathBuf>>(mut self, out_dir: P) -> Self {
        self.out_dir = out_dir.into();
        self
    }

    /// Add an extra nvcc compile argument.
    pub fn arg<S: AsRef<str>>(mut self, arg: S) -> Self {
        self.extra_args.push(arg.as_ref().to_string());
        self
    }

    /// Force the CUDA root to a specific directory.
    pub fn cuda_root<P: Into<PathBuf>>(&mut self, path: P) {
        self.cuda_root = Some(path.into());
    }

    /// Manually set the CUDA compute capability (flat code, e.g. `86` for 8.6).
    ///
    /// This overrides auto-detection from `CUDA_COMPUTE_CAP` env, cudarc, and nvidia-smi.
    pub fn compute_cap(mut self, compute_cap: usize) -> Self {
        self.compute_cap = Some(compute_cap);
        self
    }

    /// Build a static library from the kernel sources.
    ///
    /// Link with `println!("cargo:rustc-link-lib=<name>");` in your build.rs.
    pub fn build_lib<P: Into<PathBuf>>(self, out_file: P) {
        let out_file = out_file.into();
        let compute_cap = self
            .compute_cap
            .expect("No compute capability available. Set CUDA_COMPUTE_CAP env var.");
        let out_dir = self.out_dir;

        for path in &self.watch {
            println!("cargo:rerun-if-changed={}", path.display());
        }

        let cu_files: Vec<_> = self
            .kernel_paths
            .iter()
            .map(|f| {
                let mut s = DefaultHasher::new();
                f.display().to_string().hash(&mut s);
                let hash = s.finish();
                let mut obj_file = out_dir.join(format!(
                    "{}-{:x}",
                    f.file_stem()
                        .expect("kernel path should have a filename")
                        .to_string_lossy(),
                    hash
                ));
                obj_file.set_extension("o");
                (f, obj_file)
            })
            .collect();

        let out_modified: std::result::Result<_, _> =
            out_file.metadata().and_then(|m| m.modified());
        let should_compile = if let Ok(out_modified) = out_modified {
            let kernel_modified = self.kernel_paths.iter().any(|entry| {
                let in_modified = entry
                    .metadata()
                    .expect("kernel should exist")
                    .modified()
                    .expect("modified time accessible");
                in_modified.duration_since(out_modified).is_ok()
            });
            let watch_modified = self.watch.iter().any(|entry| {
                let in_modified = entry
                    .metadata()
                    .expect("watched file should exist")
                    .modified()
                    .expect("modified time accessible");
                in_modified.duration_since(out_modified).is_ok()
            });
            kernel_modified || watch_modified
        } else {
            true
        };

        let ccbin_env = std::env::var("NVCC_CCBIN");
        if should_compile {
            cu_files
                .par_iter()
                .map(|(cu_file, obj_file)| {
                    let mut command = std::process::Command::new("nvcc");
                    command
                        .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                        .arg("-c")
                        .args(["-o", obj_file.to_str().expect("valid outfile")])
                        .args(["--default-stream", "per-thread"])
                        .args(&self.extra_args);
                    if let Ok(ccbin_path) = &ccbin_env {
                        command
                            .arg("-allow-unsupported-compiler")
                            .args(["-ccbin", ccbin_path]);
                    }
                    command.arg(cu_file);
                    let output = command
                        .spawn()
                        .expect("failed spawning nvcc")
                        .wait_with_output()
                        .expect("capture nvcc output");
                    if !output.status.success() {
                        panic!(
                            "nvcc error compiling: {command:?}\n\n# stdout\n{}\n\n# stderr\n{}",
                            String::from_utf8_lossy(&output.stdout),
                            String::from_utf8_lossy(&output.stderr)
                        );
                    }
                    Ok(())
                })
                .collect::<std::result::Result<(), std::io::Error>>()
                .expect("compile kernel files");

            let obj_files: Vec<_> = cu_files.iter().map(|c| c.1.clone()).collect();
            let mut command = std::process::Command::new("nvcc");
            command
                .arg("--lib")
                .args([
                    "-o",
                    out_file.to_str().expect("library path should be valid"),
                ])
                .args(obj_files);
            let output = command
                .spawn()
                .expect("failed spawning nvcc")
                .wait_with_output()
                .expect("run nvcc linker");
            if !output.status.success() {
                panic!(
                    "nvcc link error: {command:?}\n\n# stdout\n{}\n\n# stderr\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }

    /// Build PTX files from each kernel source.
    ///
    /// Returns [`Bindings`] which can write a Rust source file with `include_str!` constants.
    /// Returns `Err` instead of panicking when compute cap or CUDA root is unavailable.
    pub fn build_ptx(self) -> Result<Bindings> {
        let cuda_root = self.cuda_root.ok_or(Error::NoCudaRoot)?;
        let compute_cap = self.compute_cap.ok_or(Error::NoComputeCap)?;
        let cuda_include_dir = cuda_root.join("include");
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            cuda_include_dir.display()
        );
        let out_dir = self.out_dir;

        let mut include_paths = self.include_paths;
        for path in &mut include_paths {
            println!("cargo:rerun-if-changed={}", path.display());
            let destination =
                out_dir.join(path.file_name().expect("include path should have filename"));
            std::fs::copy(path.clone(), destination)?;
            // Remove filename — keep just the directory
            path.pop();
        }

        include_paths.sort();
        include_paths.dedup();

        #[allow(unused)]
        let mut include_options: Vec<String> = include_paths
            .into_iter()
            .map(|s| {
                "-I".to_string()
                    + &s.into_os_string()
                        .into_string()
                        .expect("include path should be valid UTF-8")
            })
            .collect();
        include_options.push(format!("-I{}", cuda_include_dir.display()));

        let ccbin_env = std::env::var("NVCC_CCBIN");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
        for path in &self.watch {
            println!("cargo:rerun-if-changed={}", path.display());
        }

        let children: Vec<_> = self
            .kernel_paths
            .par_iter()
            .flat_map(|p| {
                println!("cargo:rerun-if-changed={}", p.display());
                let mut output = p.clone();
                output.set_extension("ptx");
                let output_filename = Path::new(&out_dir)
                    .to_path_buf()
                    .join("out")
                    .with_file_name(output.file_name().expect("kernel should have a filename"));

                let ignore = if let Ok(metadata) = output_filename.metadata() {
                    let out_modified = metadata.modified().expect("modified time accessible");
                    let in_modified = p
                        .metadata()
                        .expect("input should have metadata")
                        .modified()
                        .expect("input modified time accessible");
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    false
                };

                if ignore {
                    None
                } else {
                    let mut command = std::process::Command::new("nvcc");
                    command
                        .arg(format!("--gpu-architecture=sm_{compute_cap}"))
                        .arg("--ptx")
                        .args(["--default-stream", "per-thread"])
                        .args(["--output-directory", &out_dir.display().to_string()])
                        .args(&self.extra_args)
                        .args(&include_options);
                    if let Ok(ccbin_path) = &ccbin_env {
                        command
                            .arg("-allow-unsupported-compiler")
                            .args(["-ccbin", ccbin_path]);
                    }
                    command.arg(p);
                    Some((
                        p,
                        format!("{command:?}"),
                        command
                            .spawn()
                            .expect(
                                "nvcc failed to start. Ensure CUDA is installed and nvcc is in PATH.",
                            )
                            .wait_with_output(),
                    ))
                }
            })
            .collect();

        let ptx_paths: Vec<PathBuf> = glob::glob(&format!("{0}/**/*.ptx", out_dir.display()))
            .expect("valid glob")
            .map(|p| p.expect("valid PTX path"))
            .collect();

        // Rewrite source file if new kernels were compiled or old ones were removed
        let write = !children.is_empty() || self.kernel_paths.len() < ptx_paths.len();

        for (kernel_path, command, child) in children {
            let output = child.map_err(|e| {
                Error::CompilationFailed(format!("nvcc failed to run: {e}"))
            })?;
            if !output.status.success() {
                return Err(Error::CompilationFailed(format!(
                    "nvcc error compiling {kernel_path:?}:\n\n# CLI {command}\n\n# stdout\n{}\n\n# stderr\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
        }

        Ok(Bindings {
            write,
            paths: self.kernel_paths,
        })
    }
}

impl Bindings {
    /// Write a Rust source file with `const KERNEL_NAME: &str = include_str!(...)` for each PTX.
    pub fn write<P: AsRef<Path>>(&self, out: P) -> Result<()> {
        if self.write {
            let mut file = std::fs::File::create(&out).map_err(|e| {
                Error::Io(format!("create {}: {e}", out.as_ref().display()))
            })?;
            for kernel_path in &self.paths {
                let name = kernel_path
                    .file_stem()
                    .expect("kernel should have stem")
                    .to_str()
                    .expect("kernel path should be valid UTF-8");
                file.write_all(
                    format!(
                        r#"pub const {}: &str = include_str!(concat!(env!("OUT_DIR"), "/{}.ptx"));"#,
                        name.to_uppercase().replace('.', "_"),
                        name
                    )
                    .as_bytes(),
                )
                .map_err(|e| Error::Io(format!("write to {}: {e}", out.as_ref().display())))?;
                file.write_all(&[b'\n'])
                    .map_err(|e| Error::Io(format!("write newline: {e}")))?;
            }
        }
        Ok(())
    }
}

fn cuda_include_dir() -> Option<PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];

    #[allow(unused)]
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(std::result::Result::ok)
        .map(PathBuf::from);

    let roots = [
        "/usr",
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/CUDA",
    ];

    println!("cargo:info={roots:?}");

    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    env_vars
        .chain(roots)
        .find(|path| path.join("include").join("cuda.h").is_file())
}

/// Validate that nvcc supports the detected compute capability.
///
/// Returns `Ok(compute_cap)` if supported, or `Err` with details about what nvcc supports.
/// This is separated from detection so callers can decide whether to error or fall back.
pub fn validate_compute_cap_with_nvcc(compute_cap: usize) -> Result<usize> {
    let out = std::process::Command::new("nvcc")
        .arg("--list-gpu-code")
        .output()
        .map_err(|e| Error::DetectionFailed(format!("nvcc not found: {e}")))?;

    if !out.status.success() {
        return Err(Error::DetectionFailed(
            "nvcc --list-gpu-code failed".into(),
        ));
    }

    let stdout = std::str::from_utf8(&out.stdout)
        .map_err(|_| Error::DetectionFailed("nvcc output is not valid UTF-8".into()))?;

    let mut codes = Vec::new();
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('_').collect();
        if !parts.is_empty() && parts.contains(&"sm") {
            if let Ok(num) = parts[1].parse::<usize>() {
                codes.push(num);
            }
        }
    }
    codes.sort();

    if codes.is_empty() {
        return Err(Error::DetectionFailed(
            "no GPU codes parsed from nvcc".into(),
        ));
    }

    let max_code = *codes.last().unwrap();

    if !codes.contains(&compute_cap) {
        // If the GPU is newer than what nvcc supports, use the highest nvcc supports
        if compute_cap > max_code {
            println!(
                "cargo:warning=GPU compute cap {compute_cap} exceeds nvcc max {max_code}. \
                 Targeting {max_code} instead."
            );
            return Ok(max_code);
        }
        return Err(Error::UnsupportedComputeCap {
            requested: compute_cap,
            supported: codes,
        });
    }

    Ok(compute_cap)
}
