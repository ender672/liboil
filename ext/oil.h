#include "resample.h"
#include <stddef.h>

/* Callback to read image data.
 *
 * ctx     - context pointer, supplied at initialization
 * in_len  - requested read length
 * buf     - an already allocated buffer large enough to hold the requested read
 *           length.
 * out_len - returns the actual number of bytes read.
 *
 * Returns 0 if successful, -1 otherwise.
 *
 * The number of bytes read should be set to 0 at EOF. Otherwise must be equal
 * to or less than the requested read length.
 *
 * The callback function is allowed to longjmp. This will leave the image in an
 * incomplete and unresumable state. Just make sure to free the image struct
 * afterwards.
 */
typedef int (*read_fn_t)(void *ctx, size_t in_len, unsigned char *buf, size_t *out_len);

/* Callback to handle output image data.
 *
 * ctx_ptr - context pointer, supplied at initialization
 * buf     - buffer holding the data to be written
 * len     - buffer length
 *
 * Returns 0 if successful, -1 otherwise.
 *
 * The callback function is allowed to longjmp. This will leave the image in an
 * incomplete and unresumable state. Just make sure to free the image struct
 * afterwards.
 */
typedef int (*write_fn_t)(void *ctx, unsigned char *buf, size_t len);

/* This is the base struct for image sources and manipulations.
 *
 * width & height are the output dimensions. cmp is the number of components in
 * the output image.
 *
 * get_scanline() should populate a buffer with the next scanline of image data.
 * The caller is responsible for allocating the buffer with size [width * cmp]
 * and shall only call the function up to height times. Returns less than zero
 * on failure. The caller should not call this function again after failure.
 *
 * free() cleans up all internal memory allocations for the image source or
 * image manipulation. The caller is resposible for calling free after
 * use, even when initialization or get_scanline fails.
 */
struct image {
    long width;
    long height;
    enum sample_fmt fmt;
    int (*get_scanline)(struct image *image, unsigned char *buffer);
    void (*free)(struct image *image);
    void *data;
};

/* This is the base class for image writers.
 *
 * src is the source image for the writer.
 *
 * write() will process scanlines from src and output the results to write_fn.
 * You can optionally supply data in ctx which will be passed to write_fn.
 * write_fn is allowed to longjmp, but you must call free() afterwards.
 *
 * free() will free allocations. The caller is responsible for calling this
 * whether the write is successful or not.
 *
 * The data member is used internally.
 */
struct writer {
    struct image *src;
    int (*write)(struct writer *writer);
    void (*free)(struct writer *writer);
    write_fn_t write_cb;
    void *ctx;
    void *data;
};

/* The function oil_image_init will determine the image type based on the image
 * signature. It populates an out parameter with the type of image that was
 * detected so that the caller knows.
 */
enum image_type {
    PPM,
    JPEG,
    PNG
};

int oil_image_init(struct image *i, read_fn_t read, void *ctx, enum image_type *type);
int ext_to_image_type(char *ext, enum image_type *type);
void yscaler_init(struct image *i, struct image *src, long height);
void xscaler_init(struct image *i, struct image *src, long width);
void random_init(struct image *im, enum sample_fmt fmt, long width, long height);

int ppm_init(struct image *i, read_fn_t read, void *ctx, int sig_bytes);
int ppm_writer_init(struct writer *w, write_fn_t write, void *ctx, struct image *src);

void jpeg_appinit();
int jpeg_init(struct image *i, read_fn_t read, void *ctx, int sig_bytes);
void jpeg_set_scale_denom(struct image *i, int denom);
void jpeg_set_rgbx(struct image *i);
void jpeg_writer_init(struct writer *w, write_fn_t write, void *ctx, struct image *src);

int png_init(struct image *i, read_fn_t read, void *ctx, int sig_bytes);
void png_writer_init(struct writer *w, write_fn_t write, void *ctx, struct image *src);
