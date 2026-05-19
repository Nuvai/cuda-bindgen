use crate::error::{Error, Result};

/// CUDA compute capability (major.minor) for a device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComputeCapability {
    /// Major version (e.g. 9 for sm_90).
    pub major: i32,
    /// Minor version (e.g. 0 for sm_90).
    pub minor: i32,
}

impl ComputeCapability {
    /// Flat two-digit code (e.g. 90 for major=9, minor=0).
    pub fn as_flat(&self) -> usize {
        (self.major * 10 + self.minor) as usize
    }
}

impl std::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_flat())
    }
}

/// Queries the primary device's compute capability via the CUDA driver API.
///
/// Requires the `driver-detect` feature and a functioning NVIDIA driver.
#[cfg(feature = "driver-detect")]
pub fn get_from_driver() -> Result<ComputeCapability> {
    let ctx = cudarc::driver::CudaContext::new(0)?;
    let (major, minor) = ctx.compute_capability()?;
    Ok(ComputeCapability { major, minor })
}

/// Queries compute capability via `nvidia-smi --query-gpu=compute_cap`.
pub fn get_from_nvidia_smi() -> Result<ComputeCapability> {
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .map_err(|e| Error::DetectionFailed(format!("nvidia-smi not found: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::DetectionFailed(format!(
            "nvidia-smi failed (exit {}): {stderr}",
            output.status
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    if stdout.contains("NVIDIA-SMI has failed") {
        return Err(Error::DetectionFailed(
            "NVIDIA driver not responding (nvidia-smi reports failure)".into(),
        ));
    }

    let cap_str = stdout
        .lines()
        .next()
        .ok_or_else(|| Error::DetectionFailed("empty nvidia-smi output".into()))?;

    let parts: Vec<&str> = cap_str.trim().split('.').collect();
    if parts.len() < 2 {
        return Err(Error::DetectionFailed(format!(
            "unexpected nvidia-smi format: {cap_str:?}"
        )));
    }

    let major: i32 = parts[0]
        .parse()
        .map_err(|_| Error::DetectionFailed(format!("bad major version: {:?}", parts[0])))?;
    let minor: i32 = parts[1]
        .parse()
        .map_err(|_| Error::DetectionFailed(format!("bad minor version: {:?}", parts[1])))?;

    Ok(ComputeCapability { major, minor })
}

/// Detect compute capability using all available methods, in priority order:
///
/// 1. `CUDA_COMPUTE_CAP` environment variable (flat code, e.g. `80`)
/// 2. CUDA driver API via cudarc (if `driver-detect` feature enabled)
/// 3. `nvidia-smi` CLI fallback
///
/// Returns `Err(NoComputeCap)` if all methods fail. Never panics.
pub fn detect() -> Result<ComputeCapability> {
    if let Ok(cap_str) = std::env::var("CUDA_COMPUTE_CAP") {
        let cap_str = cap_str.trim().replace('.', "");
        if let Ok(flat) = cap_str.parse::<usize>() {
            let major = (flat / 10) as i32;
            let minor = (flat % 10) as i32;
            println!("cargo:rustc-env=CUDA_COMPUTE_CAP={flat}");
            return Ok(ComputeCapability { major, minor });
        }
    }

    #[cfg(feature = "driver-detect")]
    {
        match get_from_driver() {
            Ok(cap) => {
                println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
                return Ok(cap);
            }
            Err(e) => {
                println!("cargo:warning=cudarc driver detection failed: {e}, trying nvidia-smi");
            }
        }
    }

    match get_from_nvidia_smi() {
        Ok(cap) => {
            println!("cargo:rustc-env=CUDA_COMPUTE_CAP={cap}");
            Ok(cap)
        }
        Err(e) => {
            println!("cargo:warning=nvidia-smi detection failed: {e}");
            Err(Error::NoComputeCap)
        }
    }
}
