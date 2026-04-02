use std::ffi::CString;
use std::path::Path;

use crate::colorspace::ColorSpace;
use crate::scale::{OilError, OilScale};

// libjpeg J_COLOR_SPACE constants.
const JCS_GRAYSCALE: libc::c_int = 1;
const JCS_RGB: libc::c_int = 2;
const JCS_CMYK: libc::c_int = 4;
const JCS_EXT_RGBX: libc::c_int = 7;
const JCS_EXT_BGRX: libc::c_int = 9;

fn jcs_to_colorspace(jcs: libc::c_int) -> Result<ColorSpace, OilError> {
    match jcs {
        JCS_GRAYSCALE => Ok(ColorSpace::G),
        JCS_RGB => Ok(ColorSpace::RGB),
        JCS_CMYK => Ok(ColorSpace::CMYK),
        JCS_EXT_RGBX | JCS_EXT_BGRX => Ok(ColorSpace::RGBX),
        _ => Err(OilError::InvalidArgument),
    }
}

fn colorspace_to_jcs(cs: ColorSpace) -> libc::c_int {
    match cs {
        ColorSpace::G => JCS_GRAYSCALE,
        ColorSpace::RGB => JCS_RGB,
        ColorSpace::CMYK => JCS_CMYK,
        ColorSpace::RGBX => JCS_EXT_RGBX,
        _ => JCS_RGB,
    }
}

// Opaque handles matching the C glue typedefs.
enum OilJpegReader {}
enum OilJpegWriter {}

extern "C" {
    fn oil_jpeg_reader_create(data: *const u8, size: libc::c_ulong) -> *mut OilJpegReader;
    fn oil_jpeg_reader_create_file(path: *const libc::c_char) -> *mut OilJpegReader;
    fn oil_jpeg_dimensions_file(
        path: *const libc::c_char,
        width: *mut libc::c_uint,
        height: *mut libc::c_uint,
    ) -> libc::c_int;
    fn oil_jpeg_reader_width(r: *const OilJpegReader) -> libc::c_uint;
    fn oil_jpeg_reader_height(r: *const OilJpegReader) -> libc::c_uint;
    fn oil_jpeg_reader_color_space(r: *const OilJpegReader) -> libc::c_int;
    fn oil_jpeg_reader_read_scanline(r: *mut OilJpegReader, buf: *mut u8);
    fn oil_jpeg_reader_destroy(r: *mut OilJpegReader);

    fn oil_jpeg_writer_create(
        width: libc::c_uint,
        height: libc::c_uint,
        components: libc::c_int,
        color_space: libc::c_int,
        quality: libc::c_int,
    ) -> *mut OilJpegWriter;
    fn oil_jpeg_writer_write_scanline(w: *mut OilJpegWriter, buf: *mut u8);
    fn oil_jpeg_writer_finish(w: *mut OilJpegWriter, size: *mut libc::c_ulong) -> *mut u8;
    fn oil_jpeg_free_buf(buf: *mut u8);
}

/// RAII wrapper for the C JPEG reader handle.
struct JpegReader {
    ptr: *mut OilJpegReader,
}

impl JpegReader {
    fn new(data: &[u8]) -> Result<Self, OilError> {
        let ptr = unsafe { oil_jpeg_reader_create(data.as_ptr(), data.len() as libc::c_ulong) };
        if ptr.is_null() {
            return Err(OilError::AllocationFailed);
        }
        Ok(Self { ptr })
    }

    fn from_path(path: &Path) -> Result<Self, OilError> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| OilError::InvalidArgument)?;
        let ptr = unsafe { oil_jpeg_reader_create_file(c_path.as_ptr()) };
        if ptr.is_null() {
            return Err(OilError::InvalidArgument);
        }
        Ok(Self { ptr })
    }

    fn width(&self) -> u32 {
        unsafe { oil_jpeg_reader_width(self.ptr) }
    }

    fn height(&self) -> u32 {
        unsafe { oil_jpeg_reader_height(self.ptr) }
    }

    fn color_space(&self) -> libc::c_int {
        unsafe { oil_jpeg_reader_color_space(self.ptr) }
    }

    fn read_scanline(&mut self, buf: &mut [u8]) {
        unsafe { oil_jpeg_reader_read_scanline(self.ptr, buf.as_mut_ptr()) }
    }
}

impl Drop for JpegReader {
    fn drop(&mut self) {
        unsafe { oil_jpeg_reader_destroy(self.ptr) }
    }
}

/// RAII wrapper for the C JPEG writer handle.
struct JpegWriter {
    ptr: *mut OilJpegWriter,
}

impl JpegWriter {
    fn new(width: u32, height: u32, components: usize, color_space: libc::c_int, quality: u8) -> Result<Self, OilError> {
        let ptr = unsafe {
            oil_jpeg_writer_create(width, height, components as libc::c_int, color_space, quality as libc::c_int)
        };
        if ptr.is_null() {
            return Err(OilError::AllocationFailed);
        }
        Ok(Self { ptr })
    }

    fn write_scanline(&mut self, buf: &mut [u8]) {
        unsafe { oil_jpeg_writer_write_scanline(self.ptr, buf.as_mut_ptr()) }
    }

    fn finish(self) -> Result<Vec<u8>, OilError> {
        let mut size: libc::c_ulong = 0;
        let buf = unsafe { oil_jpeg_writer_finish(self.ptr, &mut size) };
        if buf.is_null() {
            return Err(OilError::AllocationFailed);
        }
        let result = unsafe { std::slice::from_raw_parts(buf, size as usize) }.to_vec();
        unsafe { oil_jpeg_free_buf(buf) };
        std::mem::forget(self); // writer already freed by finish()
        Ok(result)
    }
}

impl Drop for JpegWriter {
    fn drop(&mut self) {
        // finish() consumes self via forget, so this only runs on error paths
        // where finish() was never called. Clean up by finishing with a dummy.
        let mut size: libc::c_ulong = 0;
        let buf = unsafe { oil_jpeg_writer_finish(self.ptr, &mut size) };
        if !buf.is_null() {
            unsafe { oil_jpeg_free_buf(buf) };
        }
    }
}

/// Read JPEG dimensions from a file without decoding pixel data.
pub fn jpeg_dimensions_file(path: &Path) -> Result<(u32, u32), OilError> {
    let c_path = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| OilError::InvalidArgument)?;
    let mut width: libc::c_uint = 0;
    let mut height: libc::c_uint = 0;
    let ret = unsafe { oil_jpeg_dimensions_file(c_path.as_ptr(), &mut width, &mut height) };
    if ret != 0 {
        return Err(OilError::InvalidArgument);
    }
    Ok((width, height))
}

/// Decode a JPEG file, resize it using streaming scanline I/O via
/// libjpeg-turbo, and re-encode as JPEG. Reads from disk scanline-by-scanline
/// to avoid loading the entire file into memory.
pub fn resize_jpeg_file(
    path: &Path,
    out_width: u32,
    out_height: u32,
    quality: u8,
) -> Result<Vec<u8>, OilError> {
    let mut reader = JpegReader::from_path(path)?;

    let in_width = reader.width();
    let in_height = reader.height();
    let jcs = reader.color_space();
    let cs = jcs_to_colorspace(jcs)?;
    let cmp = cs.components();

    let mut scaler = OilScale::new(in_height, out_height, in_width, out_width, cs)?;

    let in_stride = in_width as usize * cmp;
    let out_stride = out_width as usize * cmp;

    let mut inbuf = vec![0u8; in_stride];
    let mut outbuf = vec![0u8; out_stride];

    let mut writer = JpegWriter::new(out_width, out_height, cmp, colorspace_to_jcs(cs), quality)?;

    for _ in 0..out_height {
        while scaler.slots() > 0 {
            reader.read_scanline(&mut inbuf);
            scaler.push_scanline(&inbuf);
        }
        scaler.read_scanline(&mut outbuf);
        writer.write_scanline(&mut outbuf);
    }

    writer.finish()
}

/// Decode a JPEG from bytes, resize it using streaming scanline I/O via
/// libjpeg-turbo, and re-encode as JPEG.
pub fn resize_jpeg(
    input: &[u8],
    out_width: u32,
    out_height: u32,
    quality: u8,
) -> Result<Vec<u8>, OilError> {
    let mut reader = JpegReader::new(input)?;

    let in_width = reader.width();
    let in_height = reader.height();
    let jcs = reader.color_space();
    let cs = jcs_to_colorspace(jcs)?;
    let cmp = cs.components();

    let mut scaler = OilScale::new(in_height, out_height, in_width, out_width, cs)?;

    let in_stride = in_width as usize * cmp;
    let out_stride = out_width as usize * cmp;

    let mut inbuf = vec![0u8; in_stride];
    let mut outbuf = vec![0u8; out_stride];

    let mut writer = JpegWriter::new(out_width, out_height, cmp, colorspace_to_jcs(cs), quality)?;

    for _ in 0..out_height {
        while scaler.slots() > 0 {
            reader.read_scanline(&mut inbuf);
            scaler.push_scanline(&inbuf);
        }
        scaler.read_scanline(&mut outbuf);
        writer.write_scanline(&mut outbuf);
    }

    writer.finish()
}
