use std::ffi::{c_int, c_void};

use cudarc::driver::sys::CUstream_st;

unsafe extern "C" {
    pub fn launch_sin2(out: *mut c_void, inp: *const c_void, n: c_int, stream: *mut CUstream_st);
}

#[cfg(test)]
mod tests {
    use cudarc::driver::{DevicePtr, DevicePtrMut, DriverError};

    #[test]
    fn test_simple() -> Result<(), DriverError> {
        let data: Vec<f32> = (0..100).map(|u| u as f32).collect();
        let ctx = cudarc::driver::CudaContext::new(0)?;
        let stream = ctx.default_stream();

        let inp = stream.clone_htod(&data)?;
        let mut out = stream.alloc_zeros::<f32>(100)?;

        {
            let (out_ptr, _out_guard) = out.device_ptr_mut(&stream);
            let out_ptr = out_ptr as *mut core::ffi::c_void;
            let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
            let inp_ptr = inp_ptr as *const core::ffi::c_void;
            unsafe { super::launch_sin2(out_ptr, inp_ptr, 100, stream.cu_stream()) };
        }

        let out_host: Vec<f32> = stream.clone_dtoh(&out)?;
        assert_eq!(out_host.len(), data.len());
        let expected: Vec<_> = data.into_iter().map(f32::sin).collect();
        for (i, (l, r)) in out_host.into_iter().zip(expected.into_iter()).enumerate() {
            let diff = (l - r).abs() / (l + 1e-10);
            assert!(diff < 1e-3, "{l} != {r} (diff = {diff:?}, location = {i})");
        }
        Ok(())
    }
}
