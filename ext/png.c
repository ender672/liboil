#include "oil.h"
#include <stdlib.h>
#include <png.h>

struct png_pair {
    png_structp png;
    png_infop info;
    unsigned char *sl_buf;
    unsigned char **scanlines;
    long ypos;
};

struct png_src {
    read_fn_t read;
    void *ctx;
};

struct png_dest {
    write_fn_t write;
    void *ctx;
};

static void warning(png_structp png_ptr, png_const_charp message) {}

static void
error(png_structp png_ptr, png_const_charp message)
{
    longjmp(png_jmpbuf(png_ptr), 1);
}

static void
read_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    struct png_src *src;
    size_t out_len;
    int ret;

    src = (struct png_src *)png_get_io_ptr(png_ptr);
    ret = src->read(src->ctx, length, data, &out_len);

    if (ret || !out_len)
	png_error(png_ptr, "IO Read Failed");
}

static void png_flush_data(png_structp png_ptr) {}

static void
write_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    struct png_dest *dest;

    dest = (struct png_dest *)png_get_io_ptr(png_ptr);
    if (dest->write(dest->ctx, data, length))
	png_error(png_ptr, "IO Write Failed");
}

static void
png_free_fn(struct image *image)
{
    struct png_pair *pair;
    struct png_src *src;
    long i;

    pair = (struct png_pair *)image->data;
    src = (struct png_src *)png_get_io_ptr(pair->png);

    png_destroy_read_struct(&pair->png, &pair->info, (png_info **)NULL);
    free(src);

    if (pair->scanlines) {
	for (i=0; i<image->height; i++)
	    free(pair->scanlines[i]);
	free(pair->scanlines);
    }

    free(pair);
}

static void
png_normalize_input(png_structp read_ptr, png_infop read_i_ptr)
{
    png_byte ctype;
    int bit_depth;

    bit_depth = png_get_bit_depth(read_ptr, read_i_ptr);
    ctype = png_get_color_type(read_ptr, read_i_ptr);

    if (ctype == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
      png_set_gray_1_2_4_to_8(read_ptr);

    if (bit_depth < 8)
       png_set_packing(read_ptr);
    else if (bit_depth == 16)
#if PNG_LIBPNG_VER >= 10504
	png_set_scale_16(read_ptr);
#else
	png_set_strip_16(read_ptr);
#endif

    if (png_get_valid(read_ptr, read_i_ptr, PNG_INFO_tRNS))
	png_set_tRNS_to_alpha(read_ptr);

    if (ctype == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(read_ptr);

    png_read_update_info(read_ptr, read_i_ptr);
}

static int
get_scanline(struct image *image, unsigned char *sl)
{
    struct png_pair *pair;

    pair = (struct png_pair *)image->data;
    if (setjmp(png_jmpbuf(pair->png)))
	return -1;
    png_read_row(pair->png, (png_bytep)sl, (png_bytep)NULL);
    return 0;
}

static int
get_scanline_interlaced(struct image *image, unsigned char *sl)
{
    struct png_pair *pair;

    pair = (struct png_pair *)image->data;
    memcpy(sl, pair->scanlines[pair->ypos], image->width * image->cmp);
    pair->ypos++;
    // TODO: could free buffers as we go along
    return 0;
}

static int
ctype_to_int(png_byte ctype)
{
    switch (ctype) {
      case PNG_COLOR_TYPE_GRAY:
	return 1;
      case PNG_COLOR_TYPE_GRAY_ALPHA:
	return 2;
      case PNG_COLOR_TYPE_RGB:
	return 3;
      case PNG_COLOR_TYPE_RGB_ALPHA:
	return 4;
    }

    return -1;
}

static int
int_to_ctype(int cmp, png_byte *ctype)
{
    switch (cmp) {
      case 1:
	*ctype = PNG_COLOR_TYPE_GRAY;
	break;
      case 2:
	*ctype = PNG_COLOR_TYPE_GRAY_ALPHA;
	break;
      case 3:
	*ctype = PNG_COLOR_TYPE_RGB;
	break;
      case 4:
	*ctype = PNG_COLOR_TYPE_RGB_ALPHA;
	break;
      default:
	return -1;
    }

    return 0;
}

int
png_init(struct image *image, read_fn_t read_cb, void *ctx, int sig_bytes)
{
    struct png_pair *pair;
    struct png_src *src;
    png_byte ctype;
    long i;

    image->free = png_free_fn;
    pair = malloc(sizeof(struct png_pair));
    pair->scanlines = NULL;
    image->data = pair;

    pair->png = png_create_read_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
				       (png_error_ptr)error,
				       (png_error_ptr)warning);

    pair->info = png_create_info_struct(pair->png);

    src = malloc(sizeof(struct png_src));
    src->read = read_cb;
    src->ctx = ctx;
    png_set_read_fn(pair->png, src, read_data);

    if (sig_bytes)
	png_set_sig_bytes(pair->png, sig_bytes);

    if (setjmp(png_jmpbuf(pair->png)))
	return -1;

    png_read_info(pair->png, pair->info);
    png_normalize_input(pair->png, pair->info);

    ctype = png_get_color_type(pair->png, pair->info);
    image->cmp = ctype_to_int(ctype);
    if (image->cmp < 0)
	return -1;

    image->width = png_get_image_width(pair->png, pair->info);
    image->height = png_get_image_height(pair->png, pair->info);

    switch (png_get_interlace_type(pair->png, pair->info)) {
      case PNG_INTERLACE_NONE:
	image->get_scanline = get_scanline;
	break;
      case PNG_INTERLACE_ADAM7:
	pair->scanlines = malloc(image->height * sizeof(png_bytep));
	for (i=0; i<image->height; i++)
	    pair->scanlines[i] = malloc(image->width * image->cmp);
	png_read_image(pair->png, pair->scanlines);
	image->get_scanline = get_scanline_interlaced;
	pair->ypos = 0;
	break;
      default:
	return -1;
    }

    return 0;
}

static void
png_writer_free(struct writer *w)
{
    struct png_pair *pair;
    struct png_dest *dest;

    pair = (struct png_pair *)w->data;
    dest = (struct png_dest *)png_get_io_ptr(pair->png);
    png_destroy_write_struct(&pair->png, &pair->info);
    free(dest);
    free(pair->sl_buf);
    free(pair);
}

static int
png_writer_write(struct writer *w)
{
    long i;
    struct png_pair *pair;
    struct image *src;

    pair = (struct png_pair *)w->data;
    src = w->src;

    if (setjmp(png_jmpbuf(pair->png)))
	return -1;

    png_write_info(pair->png, pair->info);

    for (i=0; i<src->height; i++) {
	if (src->get_scanline(src, pair->sl_buf))
	    return -1;
	png_write_row((png_structp)pair->png, (png_bytep)pair->sl_buf);
    }

    png_write_end(pair->png, pair->info);

    return 0;
}

void
png_writer_init(struct writer *w, write_fn_t write_cb, void *ctx,
		struct image *src)
{
    struct png_pair *pair;
    struct png_dest *dest;
    png_byte ctype;

    w->free = png_writer_free;
    w->write = png_writer_write;
    w->src = src;

    pair = malloc(sizeof(struct png_pair));
    pair->sl_buf = malloc(src->width * src->cmp);
    w->data = pair;

    pair->png = png_create_write_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
					(png_error_ptr)error,
					(png_error_ptr)warning);
    pair->info = png_create_info_struct(pair->png);

    dest = malloc(sizeof(struct png_dest));
    dest->write = write_cb;
    dest->ctx = ctx;

    png_set_write_fn(pair->png, dest, write_data, png_flush_data);

    if (int_to_ctype(src->cmp, &ctype))
	return;

    png_set_IHDR(pair->png, pair->info, src->width, src->height, 8, ctype,
		 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		 PNG_FILTER_TYPE_DEFAULT);

    return;
}
