fn main() {
    let builder = bindgen_cuda::Builder::default();
    let _ = builder.build_lib("libsin.a");
    println!("cargo:rustc-link-lib=sin");
    println!("cargo:rustc-link-search=native={}", ".");
    println!("cargo:rustc-link-lib=stdc++");
}
