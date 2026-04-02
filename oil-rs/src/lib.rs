pub mod colorspace;
pub mod srgb;
pub mod kernel;
pub mod scale;
pub mod jpeg;
#[cfg(feature = "ffi")]
pub mod jpeg_ffi;

pub use colorspace::ColorSpace;
pub use scale::{OilScale, OilError};
