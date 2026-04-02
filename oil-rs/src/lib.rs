pub mod colorspace;
pub mod srgb;
pub mod kernel;
pub mod scale;
pub mod jpeg;
pub mod png;
#[cfg(feature = "ffi")]
pub mod jpeg_ffi;
#[cfg(target_arch = "x86_64")]
pub mod sse2;

pub use colorspace::ColorSpace;
pub use scale::{OilScale, OilError};
