fn main() {
    #[cfg(feature = "ffi")]
    {
        cc::Build::new()
            .file("csrc/jpeg_glue.c")
            .compile("jpeg_glue");

        println!("cargo:rustc-link-lib=jpeg");
    }
}
