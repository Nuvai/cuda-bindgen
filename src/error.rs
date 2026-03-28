/// Errors that can occur during CUDA kernel building.
#[derive(Debug, Clone)]
pub enum Error {
    /// nvidia-smi or nvcc failed or returned unexpected output.
    DetectionFailed(String),
    /// Compute capability not available from any source.
    NoComputeCap,
    /// nvcc cannot target the requested compute capability.
    UnsupportedComputeCap {
        requested: usize,
        supported: Vec<usize>,
    },
    /// CUDA root directory not found.
    NoCudaRoot,
    /// PTX compilation failed.
    CompilationFailed(String),
    /// I/O error during build.
    Io(String),
    /// CUDA driver API error (only with `driver-detect` feature).
    #[cfg(feature = "driver-detect")]
    DriverError(cudarc::driver::result::DriverError),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DetectionFailed(msg) => write!(f, "CUDA detection failed: {msg}"),
            Self::NoComputeCap => write!(
                f,
                "Could not detect CUDA compute capability. \
                 Set CUDA_COMPUTE_CAP env var (e.g., 80 for Ampere) or fix nvidia-smi."
            ),
            Self::UnsupportedComputeCap {
                requested,
                supported,
            } => write!(
                f,
                "nvcc cannot target compute cap {requested}. Supported: {supported:?}"
            ),
            Self::NoCudaRoot => write!(
                f,
                "Could not find CUDA installation. Set CUDA_PATH or CUDA_ROOT env var."
            ),
            Self::CompilationFailed(msg) => write!(f, "CUDA compilation failed: {msg}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
            #[cfg(feature = "driver-detect")]
            Self::DriverError(e) => write!(f, "CUDA driver error: {e:?}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(feature = "driver-detect")]
impl From<cudarc::driver::result::DriverError> for Error {
    fn from(error: cudarc::driver::result::DriverError) -> Self {
        Self::DriverError(error)
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Convenient result alias.
pub type Result<T> = std::result::Result<T, Error>;
