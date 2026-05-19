#![deny(missing_docs)]
#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
use rayon::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Error messages
#[derive(Debug)]
pub enum Error {}

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

/// Core builder to setup the bindings options
#[derive(Debug)]
pub struct Builder {
    cuda_root: Option<PathBuf>,
    kernel_paths: Vec<PathBuf>,
    watch: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
    compute_cap: Option<usize>,
    compute_caps: Vec<usize>,
    out_dir: PathBuf,
    extra_args: Vec<&'static str>,
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
        // Use only physical cores for rayon.
        // Builds can be super consuming and exhaust resources quite fast
        // like when building flash attention kernels
        let num_cpus = std::env::var("RAYON_NUM_THREADS").map_or_else(
            |_| num_cpus::get_physical(),
            |s| usize::from_str(&s).expect("RAYON_NUM_THREADS is not set to a valid integer"),
        );

        if rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus)
            .build_global()
            .is_err()
        {
            // Already initialized, that's fine - can happen when building multiple targets
        }

        let out_dir = std::env::var("OUT_DIR").expect("Expected OUT_DIR environement variable to be present, is this running within `build.rs`?").into();

        let cuda_root = cuda_include_dir();
        let kernel_paths = default_kernels().unwrap_or_default();
        let include_paths = default_include().unwrap_or_default();
        let extra_args = vec![];
        let watch = vec![];
        let compute_cap = compute_cap().ok();
        Self {
            cuda_root,
            kernel_paths,
            watch,
            include_paths,
            extra_args,
            compute_cap,
            compute_caps: vec![],
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

/// Helper struct to create a rust file that includes compiled kernel sources.
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
    /// Setup the kernel paths. All path must be set at once and be valid files.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().kernel_paths(vec!["src/mykernel.cu"]);
    /// ```
    pub fn kernel_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Self {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        let inexistent_paths: Vec<_> = paths.iter().filter(|f| !f.exists()).collect();
        if !inexistent_paths.is_empty() {
            panic!("Kernels paths do not exist {inexistent_paths:?}");
        }
        self.kernel_paths = paths;
        self
    }

    /// Setup the paths that the lib depend on but does not need to build
    /// ```no_run
    /// let builder =
    /// bindgen_cuda::Builder::default().watch(vec!["kernels/"]);
    /// ```
    pub fn watch<T, P>(mut self, paths: T) -> Self
    where
        T: IntoIterator<Item = P>,
        P: Into<PathBuf>,
    {
        let paths: Vec<_> = paths.into_iter().map(|p| p.into()).collect();
        let inexistent_paths: Vec<_> = paths.iter().filter(|f| !f.exists()).collect();
        if !inexistent_paths.is_empty() {
            panic!("Kernels paths do not exist {inexistent_paths:?}");
        }
        self.watch = paths;
        self
    }

    /// Setup the kernel paths. All path must be set at once and be valid files.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().include_paths(vec!["src/mykernel.cuh"]);
    /// ```
    pub fn include_paths<P: Into<PathBuf>>(mut self, paths: Vec<P>) -> Self {
        self.include_paths = paths.into_iter().map(|p| p.into()).collect();
        self
    }

    /// Setup the kernels with a glob.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().kernel_paths_glob("src/**/*.cu");
    /// ```
    pub fn kernel_paths_glob(mut self, glob: &str) -> Self {
        self.kernel_paths = glob::glob(glob)
            .expect("Invalid blob")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    /// Setup the include files with a glob.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().kernel_paths_glob("src/**/*.cuh");
    /// ```
    pub fn include_paths_glob(mut self, glob: &str) -> Self {
        self.include_paths = glob::glob(glob)
            .expect("Invalid blob")
            .map(|p| p.expect("Invalid path"))
            .collect();
        self
    }

    /// Modifies the output directory.
    /// By default this is
    /// [OUT_DIR](https://doc.rust-lang.org/cargo/reference/environment-variables.html#environment-variables-cargo-sets-for-build-scripts)
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().out_dir("out/");
    /// ```
    pub fn out_dir<P: Into<PathBuf>>(mut self, out_dir: P) -> Self {
        self.out_dir = out_dir.into();
        self
    }

    /// Sets up extra nvcc compile arguments.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().arg("--expt-relaxed-constexpr");
    /// ```
    pub fn arg(mut self, arg: &'static str) -> Self {
        self.extra_args.push(arg);
        self
    }

    /// Forces the cuda root to a specific directory.
    /// By default all standard directories will be visited.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().cuda_root("/usr/local/cuda");
    /// ```
    pub fn cuda_root<P>(&mut self, path: P)
    where
        P: Into<PathBuf>,
    {
        self.cuda_root = Some(path.into());
    }

    /// Sets the CUDA compute capability manually.
    /// By default, the compute capability is detected from `nvidia-smi` or the
    /// `CUDA_COMPUTE_CAP` environment variable.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().compute_cap(86); // For RTX 3090
    /// ```
    pub fn compute_cap(mut self, compute_cap: usize) -> Self {
        self.compute_cap = Some(compute_cap);
        self
    }

    /// Sets multiple CUDA compute capabilities for multi-arch / fat binary builds.
    /// Each capability generates a separate `-gencode` flag with both PTX and SM targets.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default()
    ///     .compute_caps(vec![75, 86, 90]); // Target Turing + Ampere + Hopper
    /// ```
    pub fn compute_caps(mut self, caps: Vec<usize>) -> Self {
        self.compute_caps = caps;
        self
    }

    /// Sets the output format: PTX (default), CUBIN, or Fatbin.
    /// Fatbin bundles PTX + CUBIN for portability and performance.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default()
    ///     .output_format(bindgen_cuda::OutputFormat::Fatbin);
    /// ```
    pub fn output_format(mut self, format: OutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Enables relocatable device code (`-rdc=true`).
    /// Required for separate compilation, dynamic parallelism, and device-side linking.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().rdc(true);
    /// ```
    pub fn rdc(mut self, enable: bool) -> Self {
        self.rdc = enable;
        self
    }

    /// Enables device link-time optimization (`-dlto`). Requires CUDA 11.2+.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().dlto(true);
    /// ```
    pub fn dlto(mut self, enable: bool) -> Self {
        self.dlto = enable;
        self
    }

    /// Sets the optimization level for device code.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default()
    ///     .opt_level(bindgen_cuda::OptLevel::O3);
    /// ```
    pub fn opt_level(mut self, level: OptLevel) -> Self {
        self.opt_level = Some(level);
        self
    }

    /// Enables device debugging (`-G`). Disables optimizations on device code.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().device_debug(true);
    /// ```
    pub fn device_debug(mut self, enable: bool) -> Self {
        self.device_debug = enable;
        self
    }

    /// Enables generation of line number information (`-lineinfo`).
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().line_info(true);
    /// ```
    pub fn line_info(mut self, enable: bool) -> Self {
        self.line_info = enable;
        self
    }

    /// Sets the C++ standard version for CUDA compilation.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default()
    ///     .cuda_std(bindgen_cuda::CudaStd::Cpp17);
    /// ```
    pub fn cuda_std(mut self, std: CudaStd) -> Self {
        self.cuda_std = Some(std);
        self
    }

    /// Enables `--use_fast_math` for aggressive floating-point optimizations.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().fast_math(true);
    /// ```
    pub fn fast_math(mut self, enable: bool) -> Self {
        self.fast_math = enable;
        self
    }

    /// Adds a host compiler flag via `-Xcompiler`.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default()
    ///     .xcompiler("-fPIC")
    ///     .xcompiler("-march=native");
    /// ```
    pub fn xcompiler(mut self, flag: &str) -> Self {
        self.xcompiler_args.push(flag.to_string());
        self
    }

    /// Enables device linking (`--device-link`). Required when using RDC across translation units.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().rdc(true).device_link(true);
    /// ```
    pub fn device_link(mut self, enable: bool) -> Self {
        self.device_link = enable;
        self
    }

    /// Adds a preprocessor define (`-D`).
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default()
    ///     .define("DEBUG", None)
    ///     .define("BLOCK_SIZE", Some("256"));
    /// ```
    pub fn define(mut self, name: &str, value: Option<&str>) -> Self {
        self.defines.push((name.to_string(), value.map(|v| v.to_string())));
        self
    }

    /// Adds ptxas options (`--ptxas-options=`).
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().ptxas_options("-v");
    /// ```
    pub fn ptxas_options(mut self, opt: &str) -> Self {
        self.ptxas_options.push(opt.to_string());
        self
    }

    /// Enables resource usage reporting (`--resource-usage`).
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().resource_usage(true);
    /// ```
    pub fn resource_usage(mut self, enable: bool) -> Self {
        self.resource_usage = enable;
        self
    }

    /// Enables verbose output (`-v`).
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().verbose(true);
    /// ```
    pub fn verbose(mut self, enable: bool) -> Self {
        self.verbose = enable;
        self
    }

    /// Enables dry-run mode (`--dryrun`). Shows commands without executing.
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().dryrun(true);
    /// ```
    pub fn dryrun(mut self, enable: bool) -> Self {
        self.dryrun = enable;
        self
    }

    /// Sets the maximum register count per thread (`--maxrregcount`).
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().max_reg_count(32);
    /// ```
    pub fn max_reg_count(mut self, count: usize) -> Self {
        self.max_reg_count = Some(count);
        self
    }

    fn arch_flags(&self) -> Vec<String> {
        if !self.compute_caps.is_empty() {
            self.compute_caps
                .iter()
                .map(|cap| {
                    format!(
                        "-gencode=arch=compute_{cap},code=[sm_{cap},compute_{cap}]"
                    )
                })
                .collect()
        } else if let Some(cap) = self.compute_cap {
            vec![format!("--gpu-architecture=sm_{cap}")]
        } else {
            panic!("No compute capability set. Use compute_cap() or compute_caps().");
        }
    }

    fn apply_common_flags(&self, command: &mut std::process::Command) {
        command.args(self.arch_flags());
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
    }

    /// Consumes the builder and create a lib in the out_dir.
    /// It then needs to be linked against in your `build.rs`
    /// ```no_run
    /// let builder = bindgen_cuda::Builder::default().build_lib("libflash.a");
    /// println!("cargo:rustc-link-lib=flash");
    /// ```
    pub fn build_lib<P>(self, out_file: P)
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
                        .expect("kernels paths should include a filename")
                        .to_string_lossy(),
                    hash
                ));
                obj_file.set_extension("o");
                (f, obj_file)
            })
            .collect();
        let out_modified: Result<_, _> = out_file.metadata().and_then(|m| m.modified());
        let should_compile = if let Ok(out_modified) = out_modified {
            let kernel_modified = self.kernel_paths.iter().any(|entry| {
                let in_modified = entry
                    .metadata()
                    .expect("kernel {entry} should exist")
                    .modified()
                    .expect("kernel modified to be accessible");
                in_modified.duration_since(out_modified).is_ok()
            });
            let watch_modified = self.watch.iter().any(|entry| {
                let in_modified = entry
                    .metadata()
                    .expect("watched file {entry} should exist")
                    .modified()
                    .expect("watch modified should be accessible");
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
                self.apply_common_flags(&mut command);
                command
                    .arg("-c")
                    .args(["-o", obj_file.to_str().expect("valid outfile")]);
                if let Ok(ccbin_path) = &ccbin_env {
                    command
                        .arg("-allow-unsupported-compiler")
                        .args(["-ccbin", ccbin_path]);
                }
                command.arg(cu_file);
                let output = command
                    .spawn()
                    .expect("failed spawning nvcc")
                    .wait_with_output().expect("capture nvcc output");
                if !output.status.success() {
                    panic!(
                        "nvcc error while executing compiling: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        &command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
                }
                Ok(())
            })
            .collect::<Result<(), std::io::Error>>().expect("compile files correctly");

            if self.device_link {
                let obj_files = cu_files.iter().map(|c| c.1.clone()).collect::<Vec<_>>();
                let dlink_file = out_dir.join("dlink.o");
                let mut command = std::process::Command::new("nvcc");
                self.apply_common_flags(&mut command);
                command
                    .arg("--device-link")
                    .args(["-o", dlink_file.to_str().expect("valid dlink path")])
                    .args(&obj_files);
                let output = command
                    .spawn()
                    .expect("failed spawning nvcc")
                    .wait_with_output()
                    .expect("Run nvcc device-link");
                if !output.status.success() {
                    panic!(
                        "nvcc error while device-linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                        &command,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    )
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
                    out_file.to_str().expect("library file {out_file} to exist"),
                ])
                .args(obj_files);
            let output = command
                .spawn()
                .expect("failed spawning nvcc")
                .wait_with_output()
                .expect("Run nvcc");
            if !output.status.success() {
                panic!(
                    "nvcc error while linking: {:?}\n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                    &command,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                )
            }
        }
    }

    /// Consumes the builder and outputs compiled kernel files (PTX, CUBIN, or Fatbin
    /// depending on [`OutputFormat`]).
    ///
    /// This function returns [`Bindings`] which can then be used
    /// to create a rust source file that will include those kernels.
    /// ```no_run
    /// let bindings = bindgen_cuda::Builder::default().build_ptx().unwrap();
    /// bindings.write("src/lib.rs").unwrap();
    /// ```
    pub fn build_ptx(self) -> Result<Bindings, Error> {
        let cuda_root = self.cuda_root.clone().expect("Could not find CUDA in standard locations, set it manually using Builder().set_cuda_root(...)");
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
                out_dir.join(path.file_name().expect("include path to have filename"));
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
                        .expect("include option to be valid string")
            })
            .collect::<Vec<_>>();
        include_options.push(format!("-I{}", cuda_include_dir.display()));

        let ccbin_env = std::env::var("NVCC_CCBIN");
        println!("cargo:rerun-if-env-changed=NVCC_CCBIN");
        for path in &self.watch {
            println!("cargo:rerun-if-changed={}", path.display());
        }
        let children = self.kernel_paths
            .par_iter()
            .flat_map(|p| {
                println!("cargo:rerun-if-changed={}", p.display());
                let mut output = p.clone();
                output.set_extension(extension);
                let output_filename = std::path::Path::new(&out_dir).to_path_buf().join("out").with_file_name(output.file_name().expect("kernel to have a filename"));

                let ignore = if let Ok(metadata) = output_filename.metadata() {
                    let out_modified = metadata.modified().expect("modified to be accessible");
                    let in_modified = p.metadata().expect("input to have metadata").modified().expect("input metadata to be accessible");
                    out_modified.duration_since(in_modified).is_ok()
                } else {
                    false
                };
                if ignore {
                    None
                } else {
                    let mut command = std::process::Command::new("nvcc");
                    self.apply_common_flags(&mut command);
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
                    Some((p, format!("{command:?}"), command.spawn()
                        .expect("nvcc failed to start. Ensure that you have CUDA installed and that `nvcc` is in your PATH.").wait_with_output()))
                }
            })
            .collect::<Vec<_>>();

        let existing_paths: Vec<PathBuf> = glob::glob(&format!("{0}/**/*.{extension}", out_dir.display()))
            .expect("valid glob")
            .map(|p| p.expect("valid path"))
            .collect();
        let write = !children.is_empty() || self.kernel_paths.len() < existing_paths.len();
        for (kernel_path, command, child) in children {
            let output = child.expect("nvcc failed to run. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
            assert!(
                output.status.success(),
                "nvcc error while compiling {kernel_path:?}:\n\n# CLI {command} \n\n# stdout\n{:#}\n\n# stderr\n{:#}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Ok(Bindings {
            write,
            paths: self.kernel_paths.clone(),
            output_format,
        })
    }
}

impl Bindings {
    /// Writes a helper rust file that will include the compiled kernel sources as constants.
    /// For PTX output, constants are `&str` via `include_str!`.
    /// For CUBIN/Fatbin output, constants are `&[u8]` via `include_bytes!`.
    pub fn write<P>(&self, out: P) -> Result<(), Error>
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
                    .expect("kernel to have stem")
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
    // NOTE: copied from cudarc build.rs.
    let env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDNN_LIB",
    ];
    let env_vars = env_vars
        .into_iter()
        .map(std::env::var)
        .filter_map(Result::ok)
        .map(Into::<PathBuf>::into);

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

fn compute_cap() -> Result<usize, Error> {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");

    // Try to parse compute caps from env
    let compute_cap = if let Ok(compute_cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={compute_cap_str}");
        compute_cap_str
            .parse::<usize>()
            .expect("Could not parse code")
    } else {
        // Use nvidia-smi to get the current compute cap
        let out = std::process::Command::new("nvidia-smi")
                .arg("--query-gpu=compute_cap")
                .arg("--format=csv")
                .output()
                .expect("`nvidia-smi` failed. Ensure that you have CUDA installed and that `nvidia-smi` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("stdout is not a utf8 string");
        let mut lines = out.lines();
        assert_eq!(lines.next().expect("missing line in stdout"), "compute_cap");
        let cap = lines
            .next()
            .expect("missing line in stdout")
            .replace('.', "");
        let cap = cap.parse::<usize>().expect("cannot parse as int {cap}");
        println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
        cap
    };

    // Grab available GPU codes from nvcc and select the highest one
    let (supported_nvcc_codes, max_nvcc_code) = {
        let out = std::process::Command::new("nvcc")
                .arg("--list-gpu-code")
                .output()
                .expect("`nvcc` failed. Ensure that you have CUDA installed and that `nvcc` is in your PATH.");
        let out = std::str::from_utf8(&out.stdout).expect("valid utf-8 nvcc output");

        let out = out.lines().collect::<Vec<&str>>();
        let mut codes = Vec::with_capacity(out.len());
        for code in out {
            let code = code.split('_').collect::<Vec<&str>>();
            if !code.is_empty() && code.contains(&"sm") {
                if let Ok(num) = code[1].parse::<usize>() {
                    codes.push(num);
                }
            }
        }
        codes.sort();
        let max_nvcc_code = *codes.last().expect("no gpu codes parsed from nvcc");
        (codes, max_nvcc_code)
    };

    // Check that nvcc supports the asked compute caps
    if !supported_nvcc_codes.contains(&compute_cap) {
        panic!(
            "nvcc cannot target gpu arch {compute_cap}. Available nvcc targets are {supported_nvcc_codes:?}."
        );
    }
    if compute_cap > max_nvcc_code {
        panic!(
            "CUDA compute cap {compute_cap} is higher than the highest gpu code from nvcc {max_nvcc_code}"
        );
    }

    Ok(compute_cap)
}
