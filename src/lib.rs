#![doc = "Nuvai fork of bindgen_cuda — graceful CUDA kernel building for Rust."]

use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// CUDA compute capability detection with 3-tier fallback (env -> driver API -> nvidia-smi).
pub mod compute_cap;
/// Error types for CUDA kernel building.
pub mod error;

pub use error::Error;

/// CUDA C++ standard version for `--std` flag.
#[derive(Debug, Clone, Copy)]
pub enum CudaStd {
    /// C++03
    Cpp03,
    /// C++11
    Cpp11,
    /// C++14
    Cpp14,
    /// C++17
    Cpp17,
    /// C++20
    Cpp20,
}

impl CudaStd {
    fn as_flag(self) -> &'static str {
        match self {
            CudaStd::Cpp03 => "--std=c++03",
            CudaStd::Cpp11 => "--std=c++11",
            CudaStd::Cpp14 => "--std=c++14",
            CudaStd::Cpp17 => "--std=c++17",
            CudaStd::Cpp20 => "--std=c++20",
        }
    }
}

/// Optimization level for device code.
#[derive(Debug, Clone, Copy)]
pub enum OptLevel {
    /// No optimization (`-O0`)
    O0,
    /// Low optimization (`-O1`)
    O1,
    /// Default optimization (`-O2`)
    O2,
    /// Aggressive optimization (`-O3`)
    O3,
}

impl OptLevel {
    fn as_flag(self) -> &'static str {
        match self {
            OptLevel::O0 => "-O0",
            OptLevel::O1 => "-O1",
            OptLevel::O2 => "-O2",
            OptLevel::O3 => "-O3",
        }
    }
}

/// Output format for compiled CUDA code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// PTX assembly (default)
    Ptx,
    /// CUBIN (device binary for a specific architecture)
    Cubin,
    /// Fatbin (bundles PTX + CUBIN for portability and performance)
    Fatbin,
}

/// Default compute capabilities used when `CUDA_COMPUTE_CAPS=all`.
const DEFAULT_COMPUTE_CAPS: &[usize] = &[75, 80, 86, 89, 90];

/// Core builder for setting up CUDA kernel compilation options.
#[derive(Debug, Clone)]
pub struct Builder {
    cuda_root: Option<PathBuf>,
    kernel_paths: Vec<PathBuf>,
    watch: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
    /// Single compute capability (legacy). Ignored when `compute_caps` is set.
    compute_cap: Option<usize>,
    /// Multiple compute capabilities for fat binary / multi-gencode builds.
    compute_caps: Vec<usize>,
    out_dir: PathBuf,
    extra_args: Vec<String>,
    output_format: OutputFormat,
    rdc: bool,
    dlto: bool,
    opt_level: Option<OptLevel>,
    device_debug: bool,
    line_info: bool,
    cuda_std: Option<CudaStd>,
    fast_math: bool,
    xcompiler_args: Vec<String>,
    device_link: bool,
    defines: Vec<(String, Option<String>)>,
    ptxas_options: Vec<String>,
    resource_usage: bool,
    verbose: bool,
    dryrun: bool,
    max_reg_count: Option<usize>,
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

        // Detect compute capabilities from env vars
        let (compute_cap, compute_caps) = detect_compute_caps();

        Self {
            cuda_root,
            kernel_paths,
            watch,
            include_paths,
            extra_args,
            compute_cap,
            compute_caps: compute_caps.unwrap_or_default(),
            out_dir,
            output_format: OutputFormat::Ptx,
            rdc: false,
            dlto: false,
            opt_level: None,
            device_debug: false,
            line_info: false,
            cuda_std: None,
            fast_math: false,
            xcompiler_args: vec![],
            device_link: false,
            defines: vec![],
            ptxas_options: vec![],
            resource_usage: false,
            verbose: false,
            dryrun: false,
            max_reg_count: None,
        }
    }
}

/// Helper struct returned by [`Builder::build_ptx`]. Contains compiled kernel paths
/// and can write a Rust source file with `include_str!` / `include_bytes!` constants.
pub struct Bindings {
    write: bool,
    paths: Vec<PathBuf>,
    output_format: OutputFormat,
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
    pub fn cuda_root<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.cuda_root = Some(path.into());
        self
    }

    /// Manually set a single CUDA compute capability (flat code, e.g. `86` for 8.6).
    ///
    /// This overrides auto-detection from `CUDA_COMPUTE_CAP` env, cudarc, and nvidia-smi.
    /// For multi-architecture fat binaries, use [`compute_caps`](Self::compute_caps) instead.
    pub fn compute_cap(mut self, compute_cap: usize) -> Self {
        self.compute_cap = Some(compute_cap);
        self
    }

    /// Sets multiple CUDA compute capabilities for multi-arch / fat binary builds.
    /// Each capability generates a separate `-gencode` flag with both PTX and SM targets.
    pub fn compute_caps(mut self, caps: Vec<usize>) -> Self {
        self.compute_caps = caps;
        self
    }

    /// Sets the output format: PTX (default), CUBIN, or Fatbin.
    /// Fatbin bundles PTX + CUBIN for portability and performance.
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Enables relocatable device code (`-rdc=true`).
    /// Required for separate compilation, dynamic parallelism, and device-side linking.
    pub fn rdc(mut self, enable: bool) -> Self {
        self.rdc = enable;
        self
    }

    /// Enables device link-time optimization (`-dlto`). Requires CUDA 11.2+.
    pub fn dlto(mut self, enable: bool) -> Self {
        self.dlto = enable;
        self
    }

    /// Sets the optimization level for device code.
    pub fn opt_level(mut self, level: OptLevel) -> Self {
        self.opt_level = Some(level);
        self
    }

    /// Enables device debugging (`-G`). Disables optimizations on device code.
    pub fn device_debug(mut self, enable: bool) -> Self {
        self.device_debug = enable;
        self
    }

    /// Enables generation of line number information (`-lineinfo`).
    pub fn line_info(mut self, enable: bool) -> Self {
        self.line_info = enable;
        self
    }

    /// Sets the C++ standard version for CUDA compilation.
    pub fn cuda_std(mut self, std: CudaStd) -> Self {
        self.cuda_std = Some(std);
        self
    }

    /// Enables `--use_fast_math` for aggressive floating-point optimizations.
    pub fn fast_math(mut self, enable: bool) -> Self {
        self.fast_math = enable;
        self
    }

    /// Adds a host compiler flag via `-Xcompiler`.
    pub fn xcompiler(mut self, flag: &str) -> Self {
        self.xcompiler_args.push(flag.to_string());
        self
    }

    /// Enables device linking (`--device-link`). Required when using RDC across translation units.
    pub fn device_link(mut self, enable: bool) -> Self {
        self.device_link = enable;
        self
    }

    /// Adds a preprocessor define (`-D`).
    pub fn define(mut self, name: &str, value: Option<&str>) -> Self {
        self.defines.push((name.to_string(), value.map(|v| v.to_string())));
        self
    }

    /// Adds ptxas options (`--ptxas-options=`).
    pub fn ptxas_options(mut self, opt: &str) -> Self {
        self.ptxas_options.push(opt.to_string());
        self
    }

    /// Enables resource usage reporting (`--resource-usage`).
    pub fn resource_usage(mut self, enable: bool) -> Self {
        self.resource_usage = enable;
        self
    }

    /// Enables verbose output (`-v`).
    pub fn verbose(mut self, enable: bool) -> Self {
        self.verbose = enable;
        self
    }

    /// Enables dry-run mode (`--dryrun`). Shows commands without executing.
    pub fn dryrun(mut self, enable: bool) -> Self {
        self.dryrun = enable;
        self
    }

    /// Sets the maximum register count per thread (`--maxrregcount`).
    pub fn max_reg_count(mut self, count: usize) -> Self {
        self.max_reg_count = Some(count);
        self
    }

    /// Returns the resolved list of compute capabilities.
    /// Prefers `compute_caps` (multi-arch) over `compute_cap` (single).
    pub fn resolved_caps(&self) -> error::Result<Vec<usize>> {
        if !self.compute_caps.is_empty() {
            Ok(self.compute_caps.clone())
        } else if let Some(cap) = self.compute_cap {
            Ok(vec![cap])
        } else {
            Err(Error::NoComputeCap)
        }
    }

    fn arch_flags(&self) -> error::Result<Vec<String>> {
        let caps = self.resolved_caps()?;
        if caps.len() > 1 {
            Ok(caps
                .iter()
                .map(|cap| {
                    format!(
                        "-gencode=arch=compute_{cap},code=[sm_{cap},compute_{cap}]"
                    )
                })
                .collect())
        } else {
            Ok(vec![format!("--gpu-architecture=sm_{}", caps[0])])
        }
    }

    fn apply_common_flags(&self, command: &mut std::process::Command) -> error::Result<()> {
        #[cfg(windows)]
        command.args(["-Xcompiler", "/Zc:preprocessor", "-DNOGDI"]);

        command.args(self.arch_flags()?);
        command.args(["--default-stream", "per-thread"]);

        if self.rdc {
            command.arg("-rdc=true");
        }
        if self.dlto {
            command.arg("-dlto");
        }
        if let Some(level) = self.opt_level {
            command.arg(level.as_flag());
        }
        if self.device_debug {
            command.arg("-G");
        }
        if self.line_info {
            command.arg("-lineinfo");
        }
        if let Some(std) = self.cuda_std {
            command.arg(std.as_flag());
        }
        if self.fast_math {
            command.arg("--use_fast_math");
        }
        for flag in &self.xcompiler_args {
            command.arg(format!("-Xcompiler={flag}"));
        }
        for (name, value) in &self.defines {
            match value {
                Some(v) => command.arg(format!("-D{name}={v}")),
                None => command.arg(format!("-D{name}")),
            };
        }
        for opt in &self.ptxas_options {
            command.arg(format!("--ptxas-options={opt}"));
        }
        if self.resource_usage {
            command.arg("--resource-usage");
        }
        if self.verbose {
            command.arg("-v");
        }
        if self.dryrun {
            command.arg("--dryrun");
        }
        if let Some(count) = self.max_reg_count {
            command.arg(format!("--maxrregcount={count}"));
        }

        command.args(&self.extra_args);
        Ok(())
    }

    /// Build a static library from the CUDA kernel files.
    ///
    /// When multiple compute capabilities are configured (via [`compute_caps`](Self::compute_caps)
    /// or `CUDA_COMPUTE_CAPS`), uses `-gencode` flags for fat binary builds.
    ///
    /// Link with `println!("cargo:rustc-link-lib=<name>");` in your build.rs.
    /// Returns `Err` instead of panicking when compute cap is unavailable or compilation fails.
    pub fn build_lib<P>(&self, out_file: P) -> error::Result<()>
    where
        P: Into<PathBuf>,
    {
        let out_file = out_file.into();
        let out_dir = self.out_dir.clone();

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
            let compile_errors: Vec<_> = cu_files
            .par_iter()
            .filter_map(|(cu_file, obj_file)| {
                let mut command = std::process::Command::new("nvcc");
                if let Err(e) = self.apply_common_flags(&mut command) {
                    return Some(format!("flag error: {e}"));
                }
                command
                    .arg("-c")
                    .args(["-o", obj_file.to_str().expect("valid outfile")]);
                if let Ok(ccbin_path) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin_path]);
                }
                command.arg(cu_file);
                let output = match command.spawn().and_then(|c| c.wait_with_output()) {
                    Ok(o) => o,
                    Err(e) => return Some(format!("nvcc failed to start: {e}")),
                };
                if !output.status.success() {
                    return Some(format!(
                        "nvcc error compiling {:?}:\n# stdout\n{}\n# stderr\n{}",
                        cu_file,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    ));
                }
                None
            })
            .collect();

            if !compile_errors.is_empty() {
                return Err(Error::CompilationFailed(compile_errors.join("\n\n")));
            }

            if self.device_link {
                let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
                let dlink_file = out_dir.join("dlink.o");
                let mut command = std::process::Command::new("nvcc");
                self.apply_common_flags(&mut command)?;
                command
                    .arg("--device-link")
                    .args(["-o", dlink_file.to_str().expect("valid dlink path")])
                    .args(&obj_files);
                let output = command.spawn()
                    .and_then(|c| c.wait_with_output())
                    .map_err(|e| Error::CompilationFailed(format!("nvcc device-link failed to start: {e}")))?;
                if !output.status.success() {
                    return Err(Error::CompilationFailed(format!(
                        "nvcc device-link error:\n# stdout\n{}\n# stderr\n{}",
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )));
                }
            }

            let mut obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
            if self.device_link {
                obj_files.push(out_dir.join("dlink.o"));
            }

            let mut command = std::process::Command::new("nvcc");
            command
                .arg("--lib")
                .args([
                    "-o",
                    out_file.to_str().expect("valid library output path"),
                ])
                .args(obj_files);
            let output = command.spawn()
                .and_then(|c| c.wait_with_output())
                .map_err(|e| Error::CompilationFailed(format!("nvcc lib failed to start: {e}")))?;
            if !output.status.success() {
                return Err(Error::CompilationFailed(format!(
                    "nvcc lib error:\n# stdout\n{}\n# stderr\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
        }
        Ok(())
    }

    /// Build compiled kernel files (PTX, CUBIN, or Fatbin depending on [`OutputFormat`]).
    ///
    /// When multiple compute capabilities are configured, compiles for the lowest CC
    /// since PTX is forward-compatible via JIT compilation on newer GPUs.
    ///
    /// Returns [`Bindings`] which can write a Rust source file with `include_str!` / `include_bytes!` constants.
    /// Returns `Err` instead of panicking when compute cap or CUDA root is unavailable.
    pub fn build_ptx(&self) -> error::Result<Bindings> {
        let cuda_root = self.cuda_root.clone().ok_or(Error::NoCudaRoot)?;
        let cuda_include_dir = cuda_root.join("include");
        println!(
            "cargo:rustc-env=CUDA_INCLUDE_DIR={}",
            cuda_include_dir.display()
        );
        let out_dir = self.out_dir.clone();
        let output_format = self.output_format;

        let (format_flag, extension) = match output_format {
            OutputFormat::Ptx => ("--ptx", "ptx"),
            OutputFormat::Cubin => ("--cubin", "cubin"),
            OutputFormat::Fatbin => ("--fatbin", "fatbin"),
        };

        let mut include_paths = self.include_paths.clone();
        for path in &mut include_paths {
            println!("cargo:rerun-if-changed={}", path.display());
            let destination =
                out_dir.join(path.file_name().expect("include path should have filename"));
            std::fs::copy(path.clone(), destination).expect("copy include headers");
            path.pop();
        }

        include_paths.sort();
        include_paths.dedup();

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
                output.set_extension(extension);
                let output_filename = std::path::Path::new(&out_dir).to_path_buf().join("out").with_file_name(output.file_name().expect("kernel to have a filename"));

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
                    if let Err(e) = self.apply_common_flags(&mut command) {
                        return Some(Err(e));
                    }
                    command
                        .arg(format_flag)
                        .args(["--output-directory", &out_dir.display().to_string()])
                        .args(&include_options);
                    if let Ok(ccbin_path) = &ccbin_env {
                        command
                            .arg("-allow-unsupported-compiler")
                            .args(["-ccbin", ccbin_path]);
                    }
                    command.arg(p);
                    let result = command.spawn()
                        .and_then(|c| c.wait_with_output());
                    Some(Ok((p, format!("{command:?}"), result)))
                }
            })
            .collect();

        let existing_paths: Vec<PathBuf> = glob::glob(&format!("{0}/**/*.{extension}", out_dir.display()))
            .into_iter()
            .flatten()
            .filter_map(|p| p.ok())
            .collect();
        let write = !children.is_empty() || self.kernel_paths.len() < existing_paths.len();
        for item in children {
            let (kernel_path, command, child) = item?;
            let output = child.map_err(|e| {
                Error::CompilationFailed(format!("nvcc failed to run: {e}"))
            })?;
            if !output.status.success() {
                return Err(Error::CompilationFailed(format!(
                    "nvcc error compiling {kernel_path:?}:\n# CLI {command}\n# stdout\n{}\n# stderr\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )));
            }
        }

        Ok(Bindings {
            write,
            paths: self.kernel_paths.clone(),
            output_format,
        })
    }
}

impl Bindings {
    /// Writes a helper Rust file that includes compiled kernel sources as constants.
    /// For PTX output, constants are `&str` via `include_str!`.
    /// For CUBIN/Fatbin output, constants are `&[u8]` via `include_bytes!`.
    pub fn write<P>(&self, out: P) -> error::Result<()>
    where
        P: AsRef<Path>,
    {
        if self.write {
            let ext = match self.output_format {
                OutputFormat::Ptx => "ptx",
                OutputFormat::Cubin => "cubin",
                OutputFormat::Fatbin => "fatbin",
            };
            let use_bytes = matches!(self.output_format, OutputFormat::Cubin | OutputFormat::Fatbin);
            let mut file = std::fs::File::create(out).expect("Create lib in {out}");
            for kernel_path in &self.paths {
                let name = kernel_path
                    .file_stem()
                    .expect("kernel should have stem")
                    .to_str()
                    .expect("kernel path to be valid");
                let const_name = name.to_uppercase().replace('.', "_");
                let line = if use_bytes {
                    format!(
                        r#"pub const {const_name}: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/{name}.{ext}"));"#,
                    )
                } else {
                    format!(
                        r#"pub const {const_name}: &str = include_str!(concat!(env!("OUT_DIR"), "/{name}.{ext}"));"#,
                    )
                };
                file.write_all(line.as_bytes()).expect("write to {out}");
                file.write_all(&[b'\n']).expect("write to {out}");
            }
        }
        Ok(())
    }
}

fn cuda_include_dir() -> Option<PathBuf> {
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_HOME",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];

    for var in &env_vars {
        println!("cargo:rerun-if-env-changed={var}");
    }

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
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
        "C:/Program Files/NVIDIA GPU Computing Toolkit",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
        "C:/CUDA",
    ];

    let roots = roots.into_iter().map(Into::<PathBuf>::into);

    // Also check versioned /usr/local/cuda-* directories
    let versioned = glob::glob("/usr/local/cuda-*")
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|p| p.ok());

    env_vars
        .chain(roots)
        .chain(versioned)
        .find(|path| path.join("include").join("cuda.h").is_file())
}

/// Detect compute capabilities from environment variables, with fallback to single-cap detection.
///
/// Priority:
/// 1. `CUDA_COMPUTE_CAPS` (plural, comma-separated, or `all` for default set)
/// 2. `CUDA_COMPUTE_CAP` / cudarc / nvidia-smi (single cap, via `compute_cap::detect`)
///
/// Returns `(Option<single>, Option<multi>)`.
fn detect_compute_caps() -> (Option<usize>, Option<Vec<usize>>) {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAPS");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // 1. Check CUDA_COMPUTE_CAPS (plural) first
    if let Ok(caps_str) = std::env::var("CUDA_COMPUTE_CAPS") {
        let caps_str = caps_str.trim();
        if caps_str.eq_ignore_ascii_case("all") {
            let caps = DEFAULT_COMPUTE_CAPS.to_vec();
            println!(
                "cargo:rustc-env=CUDA_COMPUTE_CAPS={}",
                caps.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
            );
            return (None, Some(caps));
        }

        let caps: Vec<usize> = caps_str
            .split(',')
            .filter_map(|s| {
                let s = s.trim().replace('.', "");
                compute_cap::parse_compute_cap_digits(&s)
            })
            .collect();

        if !caps.is_empty() {
            let mut caps = caps;
            caps.sort();
            caps.dedup();
            println!(
                "cargo:rustc-env=CUDA_COMPUTE_CAPS={}",
                caps.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(",")
            );
            return (None, Some(caps));
        }
    }

    // 2. Fallback to single compute cap detection
    let compute_cap = compute_cap::detect().ok().map(|c| c.as_flat());
    (compute_cap, None)
}

/// Validate that nvcc supports the given compute capability.
///
/// Returns the validated cap, or downgrades if the GPU is newer than nvcc supports.
/// This is separated from detection so callers can decide whether to error or fall back.
pub fn validate_compute_cap_with_nvcc(compute_cap: usize) -> error::Result<usize> {
    let out = std::process::Command::new("nvcc")
        .arg("--list-gpu-code")
        .output()
        .map_err(|e| Error::DetectionFailed(format!("nvcc not found: {e}")))?;

    if !out.status.success() {
        return Err(Error::DetectionFailed("nvcc --list-gpu-code failed".into()));
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
        return Err(Error::DetectionFailed("no GPU codes parsed from nvcc".into()));
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
