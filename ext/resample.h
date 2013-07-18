/**
 * The format in which image samples are stored. The RGBX format is an
 * optimization to allow memory-aligned access to sample data on an RGB image.
 */
enum sample_fmt {
    SAMPLE_GREYSCALE,
    SAMPLE_GREYSCALE_ALPHA,
    SAMPLE_RGB,
    SAMPLE_RGBA,
    SAMPLE_RGBX,
    SAMPLE_UNKNOWN
};

int sample_size(enum sample_fmt fmt);

/**
 * Scanline with extra space at the beginning and end. This allows us to extend
 * a scanline to the left and right. This in turn allows resizing functions
 * to operate past the edges of the scanline without having to check for
 * boundaries.
 */
struct padded_sl {
    void *buf;
    void *pad_left;
    long in_width;
    long out_width;
    enum sample_fmt fmt;
};

void padded_sl_init(struct padded_sl *psl, long in_width, long out_width, enum sample_fmt fmt);
void padded_sl_scale(struct padded_sl *psl, unsigned char *out);
void padded_sl_free(struct padded_sl *psl);

/**
 * Array of scanlines, forming a strip. The virtual array allows us to advance
 * the strip beyond the top & bottom. This is needed by some resamplers.
 */
struct strip {
    long in_lineno;
    long out_lineno;
    long in_height;
    long out_height;
    void **sl;
    void **virt;
    long width;
    enum sample_fmt fmt;
};

void strip_init(struct strip *st, long in_height, long out_height, long width, enum sample_fmt fmt);

/**
 * Ask the strip for a buffer that needs to be filled. Returns a pointer to an
 * allocated buffer of size <width> * <sample width>. The sample width depends
 * on the sample format.
 *
 * The buffer can then be filled with source image data, and strip_next_inbuf()
 * can then be called again.
 *
 * This method must be called in a loop until it returns NULL. Then it is safe
 * to scale the strip.
 */
void *strip_next_inbuf(struct strip *st);
void strip_scale(struct strip *st, void *sl_out);
void strip_free(struct strip *st);
