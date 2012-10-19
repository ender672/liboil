#include <ruby.h>
#include <jpeglib.h>
#include <png.h>

#define BUF_LEN 8192

/* Number of bytes we read to detect the file signature */
#define SIG_LEN 2

/* Map from a discreet dest coordinate to a discreet source coordinate.
 * The resulting coordinate can range from -0.5 to the maximum of the
 * destination image dimension.
 */
#define MAP(i, scale) (i + 0.5) / scale - 0.5;

/* jpeg signatures */
static unsigned char jpeg_sig[SIG_LEN];
static unsigned char jpeg_eof[2];

static ID id_read, id_seek;
static VALUE sym_exact;

struct thumbdata {
    VALUE io;
    long width;
    long height;
    int precision;
};

static struct jpeg_error_mgr jerr;

struct jpeg_dest {
    struct jpeg_destination_mgr mgr;
    VALUE buffer;
};

struct jpeg_src {
    struct jpeg_source_mgr mgr;
    VALUE io;
    VALUE buffer;
};

struct png_src {
    VALUE io;
    VALUE buffer;
};

struct bitmap {
    long rowlen;
    char *cur;
};

struct reader {
    long width;
    long height;
    int cmp;
    void (*read)(char *, void *);
    void *data;
};

struct writer {
    long width;
    long height;
    void (*write)(char *, void *);
    void *data;
};

struct strip {
    struct reader *reader;
    int height;
    long pos;
    unsigned char **sl;
    unsigned char **virt;
};

enum {
    RESIZE_FAST,
    RESIZE_EXACT
};

/* bitmap source callback */

static void
bitmap_read(char *sl, void *data)
{
    struct bitmap *b = (struct bitmap *)data;
    memcpy(sl, b->cur, b->rowlen);
    b->cur += b->rowlen;
}

/* jpeglib error callbacks */

static void output_message(j_common_ptr cinfo) {}

static void
error_exit(j_common_ptr cinfo)
{
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message) (cinfo, buffer);
    rb_raise(rb_eRuntimeError, "jpeglib: %s", buffer);
}

/* jpeglib source callbacks */

static void term_source(j_decompress_ptr cinfo){};

static void init_source(j_decompress_ptr cinfo)
{
    struct jpeg_src *src = (struct jpeg_src *)cinfo->src;
    src->mgr.next_input_byte = jpeg_sig;
    src->mgr.bytes_in_buffer = 2;
};

static boolean
fill_input_buffer(j_decompress_ptr cinfo)
{
    struct jpeg_src *src = (struct jpeg_src *)cinfo->src;
    VALUE ret, string = src->buffer;
    char *buf;

    ret = rb_funcall(src->io, id_read, 2, INT2FIX(BUF_LEN), string);
    src->mgr.bytes_in_buffer = RSTRING_LEN(string);
    src->mgr.next_input_byte = (JOCTET *)RSTRING_PTR(string);

    if (!src->mgr.bytes_in_buffer || NIL_P(ret)) {
	src->mgr.next_input_byte = jpeg_eof;
	src->mgr.bytes_in_buffer = 2;
    }

    return TRUE;
}

static void
skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
    struct jpeg_src *src = (struct jpeg_src *)cinfo->src;

    if (num_bytes > (long)src->mgr.bytes_in_buffer) {
	num_bytes -= (long)src->mgr.bytes_in_buffer;
	rb_funcall(src->io, id_seek, 2, INT2FIX(num_bytes), INT2FIX(SEEK_CUR));
	src->mgr.bytes_in_buffer = 0;
    } else {
	src->mgr.next_input_byte += (size_t)num_bytes;
	src->mgr.bytes_in_buffer -= (size_t)num_bytes;
    }
}

/* jpeglib destination callbacks */

static void init_destination(j_compress_ptr cinfo) {}

static boolean
empty_output_buffer(j_compress_ptr cinfo)
{
    struct jpeg_dest *dest = (struct jpeg_dest *)cinfo->dest;
    rb_yield(dest->buffer);
    dest->mgr.next_output_byte = RSTRING_PTR(dest->buffer);
    dest->mgr.free_in_buffer = RSTRING_LEN(dest->buffer);
    return TRUE;
}

static void
term_destination(j_compress_ptr cinfo)
{
    struct jpeg_dest *dest = (struct jpeg_dest *)cinfo->dest;
    size_t len = BUF_LEN - dest->mgr.free_in_buffer;

    if (len) {
	rb_str_resize(dest->buffer, len);
	rb_yield(dest->buffer);
    }
}

/* libpng error callbacks */

static void png_warning_fn(png_structp png_ptr, png_const_charp message) {}

static void
png_error_fn(png_structp png_ptr, png_const_charp message)
{
    rb_raise(rb_eRuntimeError, "pnglib: %s", message);
}

/* libpng io callbacks */

void png_flush_data(png_structp png_ptr){}

void
png_write_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    if (!png_ptr) return;
    rb_yield(rb_str_new(data, length));
}

void
png_read_data(png_structp png_ptr, png_bytep data, png_size_t length)
{
    VALUE ret, buf;
    png_size_t rlen;
    struct png_src *io_ptr;

    if (!png_ptr) return;
    io_ptr = (struct png_src *)png_get_io_ptr(png_ptr);
    buf = io_ptr->buffer;
    ret = rb_funcall(io_ptr->io, id_read, 2, INT2FIX(length), buf);

    rlen = RSTRING_LEN(buf);
    if (rlen > length)
	rb_raise(rb_eRuntimeError, "IO return buffer is too big.");
    if (!NIL_P(ret)) memcpy(data, RSTRING_PTR(buf), rlen);
}

/* png read/write callbacks */

static void
png_read(char *sl, void *data)
{
    png_read_row((png_structp)data, (png_bytep)sl, (png_bytep)NULL);
}

static void
png_write(char *sl, void *data)
{
    png_write_row((png_structp)data, (png_bytep)sl);
}

/* strip - A virtual band of scanlines */

static void
step_strip(struct strip *strip)
{
    long pos;
    int i, sl_pos;
    unsigned char **sl, **virt;

    virt = strip->virt;
    sl = strip->sl;
    pos = strip->pos;
    sl_pos = pos%strip->height;

    for (i=1; i<strip->height; i++)
	virt[i - 1] = virt[i];

    if (pos < strip->reader->height) {
	strip->reader->read(sl[sl_pos], strip->reader->data);
	virt[strip->height - 1] = sl[sl_pos];
    }

    strip->pos++;
}

static void
move_strip(struct strip *strip, double fpos)
{
    long p, target_pos;

    p = fpos < 0 ? -1 : (long)fpos;
    target_pos = p + strip->height / 2 + 1;

    while (strip->pos < target_pos)
	step_strip(strip);
}

static void
init_strip(struct strip *strip, struct reader *reader, int height)
{
    int i;
    long sl_width;

    strip->reader = reader;
    strip->height = height;
    strip->sl = malloc(height * sizeof(unsigned char*));
    strip->virt = malloc(height * sizeof(unsigned char *));
    strip->pos = 0;

    sl_width = reader->width * reader->cmp;
    for (i=0; i<height; i++) {
	strip->sl[i] = malloc(sl_width);
	strip->virt[i] = strip->sl[0];
    }
}

static void
free_strip(struct strip *strip)
{
    int i;

    for (i=0; i<strip->height; i++)
	free(strip->sl[i]);
    free(strip->sl);
    free(strip->virt);
}

/* bicubic resizer */

static double
catrom(double x)
{
    if (x<1) return (9*x*x*x - 15*x*x + 6) / 6;
    return (-3*x*x*x + 15*x*x - 24*x + 12) / 6;
}

#define BOUND(x, l, r) x = x <= l ? l : x >= r ? r : x;

static void
bicubic_x(double *in, unsigned char *out, long sw, long dw, int cmp)
{
    double scale, xsmp, tx, p1, p2, p3, p4, result, *c1, *c2, *c3, *c4, *right;
    long xsmp_i, o1, o2, o3, o4, i, j;

    scale = dw / (double)sw;

    for (i = 0; i < dw; i++) {
	xsmp = MAP(i, scale);
	xsmp_i = xsmp < 0 ? -1 : (long)xsmp;
	tx = xsmp - xsmp_i;

	p1 = catrom(1 + tx);
	p2 = catrom(tx);
	p3 = catrom(1 - tx);
	p4 = catrom(2 - tx);

	c2 = in + xsmp_i * cmp;
	c1 = c2 - cmp;
	c3 = c2 + cmp;
	c4 = c3 + cmp;
	right = in + (sw - 1) * cmp;
	BOUND(c1, in, right);
	BOUND(c2, in, right);
	BOUND(c3, in, right);
	BOUND(c4, in, right);

	for (j = 0; j < cmp; j++) {
	    result = p1 * c1[j] + p2 * c2[j] + p3 * c3[j] + p4 * c4[j];
	    if (result < 0) result = 0;
	    if (result > 255) result = 255;
	    *out++ = result;
	}
    }
}

static void
bicubic_y(struct strip *strip, double *out, double ysmp)
{
    double ty, p1, p2, p3, p4;
    unsigned char **in;
    long width, i;

    in = strip->virt;
    width = strip->reader->width * strip->reader->cmp;
    ty = ysmp - (ysmp < 0 ? -1 : (long)ysmp);
    p4 = catrom(2 - ty);
    p3 = catrom(1 - ty);
    p2 = catrom(ty);
    p1 = catrom(1 + ty);

    for (i = 0; i < width; i++) {
	out[i] = p1 * in[0][i] + p2 * in[1][i] + p3 * in[2][i] + p4 * in[3][i];
    }
}

static VALUE
bicubic2(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    struct strip *strip=(struct strip *)args[0];
    struct writer *writer=(struct writer *)args[1];
    double *sl_tmp=(double *)args[2];
    unsigned char *sl_out=(unsigned char *)args[3];
    unsigned char *sl_virt[4];
    struct reader *reader;
    double ysmp, yscale;
    long i, in_sl_width;

    reader = strip->reader;
    yscale = writer->height / (double)reader->height;
    in_sl_width = reader->width * reader->cmp;

    for (i = 0; i<writer->height; i++) {
	ysmp = MAP(i, yscale);
	move_strip(strip, ysmp);
	bicubic_y(strip, sl_tmp, ysmp);
	bicubic_x(sl_tmp, sl_out, reader->width, writer->width, reader->cmp);
	writer->write(sl_out, writer->data);
    }

    return Qnil;
}

static void
bicubic(struct reader *reader, struct writer *writer)
{
    VALUE args[4];
    int state;
    long in_len;
    struct strip strip;
    unsigned char *sl_out;
    double *sl_tmp; /* TODO: investigate performance when using an unsigned char */

    init_strip(&strip, reader, 4);
    in_len = (reader->width + 4) * reader->cmp;
    sl_tmp = malloc(in_len * sizeof(double));
    sl_out = malloc(writer->width * reader->cmp);

    args[0] = (VALUE)&strip;
    args[1] = (VALUE)writer;
    args[2] = (VALUE)sl_tmp;
    args[3] = (VALUE)sl_out;
    rb_protect(bicubic2, (VALUE)args, &state);

    free(sl_tmp);
    free(sl_out);
    free_strip(&strip);

    if (state) rb_jump_tag(state);
}

/* bilinear resizer */

static void
bilinear3(struct strip *strip, unsigned char *sl_out, double ysmp, long dw)
{
    double xsmp, scale, tx, ty, p1, p2, p3, p4;
    unsigned char **sl, *c1, *c2, *c3, *c4;
    long xsmp_i, i, j, sw;
    int cmp;

    cmp = strip->reader->cmp;
    sw = strip->reader->width;
    sl = strip->virt;
    scale = dw / (double)sw;
    ty = ysmp - (ysmp < 0 ? -1 : (long)ysmp);

    for (i = 0; i < dw; i++) {
	xsmp = MAP(i, scale);
	xsmp_i = xsmp < 0 ? -1 : (long)xsmp;
	tx = xsmp - xsmp_i;

	p4 =  tx * ty;
	p3 = (1 - tx) * ty;
	p2 =  tx - p4;
	p1 = (1 - tx) - p3;

	c1 = c2 = sl[0] + xsmp_i * cmp;
	c3 = c4 = sl[1] + xsmp_i * cmp;

	if (xsmp_i < 0) {
	    c1 = c2 = sl[0];
	    c3 = c4 = sl[1];
	} else if (xsmp_i < sw - 1) {
	    c2 += cmp;
	    c4 += cmp;
	}

	for (j = 0; j < cmp; j++)
	    *sl_out++ = p1 * c1[j] + p2 * c2[j] + p3 * c3[j] + p4 * c4[j];
    }
}

static VALUE
bilinear2(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    struct strip *strip=(struct strip *)args[0];
    struct writer *writer=(struct writer *)args[1];
    unsigned char *sl_out=(char *)args[2];
    long i;
    double ysmp, yscale;

    yscale = writer->height / (double)strip->reader->height;

    for (i = 0; i<writer->height; i++) {
	ysmp = MAP(i, yscale);
	move_strip(strip, ysmp);
	bilinear3(strip, sl_out, ysmp, writer->width);
	writer->write(sl_out, writer->data);
    }

    return Qnil;
}

static void
bilinear(struct reader *reader, struct writer *writer)
{
    VALUE args[3];
    int state;
    struct strip strip;
    unsigned char *sl_out;

    init_strip(&strip, reader, 2);
    sl_out = malloc(writer->width * reader->cmp);

    args[0] = (VALUE)&strip;
    args[1] = (VALUE)writer;
    args[2] = (VALUE)sl_out;
    rb_protect(bilinear2, (VALUE)args, &state);

    free_strip(&strip);
    free(sl_out);
 
    if (state)
	rb_jump_tag(state);
}

/* helper functions */

static void
resize(struct reader *reader, struct writer *writer, int precision)
{
    if (precision == RESIZE_FAST)
	bilinear(reader, writer);
    else
	bicubic(reader, writer);
}

static void
fix_ratio(long sw, long sh, long *dw, long *dh)
{
    double x, y;
    x = *dw / (float)sw;
    y = *dh / (float)sh;
    if (x<y) *dh = sh * x;
    else *dw = sw * y;
    if (!*dh) *dh = 1;
    if (!*dw) *dw = 1;
}

/* jpeg helper functions */

static void
jpeg_set_output_header(j_compress_ptr cinfo, j_decompress_ptr dinfo)
{
    cinfo->input_components = dinfo->output_components;
    cinfo->in_color_space = dinfo->out_color_space;
    jpeg_set_defaults(cinfo);
}

static void
jpeg_pre_scale(j_compress_ptr cinfo, j_decompress_ptr dinfo)
{
    int inv_scale = dinfo->output_width / cinfo->image_width;

    if (inv_scale >= 8) dinfo->scale_denom = 8;
    else if (inv_scale >= 4) dinfo->scale_denom = 4;
    else if (inv_scale >= 2) dinfo->scale_denom = 2;
    jpeg_calc_output_dimensions(dinfo);
}

static void
jpeg_read(char *sl, void *data)
{
    jpeg_read_scanlines((j_decompress_ptr)data, (JSAMPROW *)&sl, 1);
}

static void
jpeg_write(char *sl, void *data)
{
    jpeg_write_scanlines((j_compress_ptr)data, (JSAMPROW *)&sl, 1);
}

static VALUE
jpeg_each3(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    j_decompress_ptr dinfo = (j_decompress_ptr)args[0];
    j_compress_ptr cinfo = (j_compress_ptr)args[1];
    struct thumbdata *thumb = (struct thumbdata *)args[2];
    struct reader reader;
    struct writer writer;
    long dest_width, dest_height;

    jpeg_read_header(dinfo, TRUE);
    jpeg_calc_output_dimensions(dinfo);
    jpeg_set_output_header(cinfo, dinfo);

    dest_width = cinfo->image_width;
    dest_height = cinfo->image_height;
    fix_ratio(dinfo->output_width, dinfo->output_height, &dest_width,
    	      &dest_height);
    cinfo->image_width = dest_width;
    cinfo->image_height = dest_height;
    jpeg_pre_scale(cinfo, dinfo);

    jpeg_start_compress(cinfo, TRUE);
    jpeg_start_decompress(dinfo);

    reader.width = dinfo->output_width;
    reader.height = dinfo->output_height;
    reader.cmp = dinfo->output_components;
    reader.read = jpeg_read;
    reader.data = dinfo;

    writer.width = cinfo->image_width;
    writer.height = cinfo->image_height;
    writer.write = jpeg_write;
    writer.data = cinfo;

    resize(&reader, &writer, thumb->precision);

    jpeg_abort_decompress(dinfo);
    jpeg_finish_compress(cinfo);

    return Qnil;
}

static void
jpeg_each2(struct thumbdata *data, struct jpeg_src *src, struct jpeg_dest *dest)
{
    int state;
    VALUE args[3];
    struct jpeg_decompress_struct dinfo;
    struct jpeg_compress_struct cinfo;

    dinfo.err = cinfo.err = &jerr;

    jpeg_create_compress(&cinfo);
    cinfo.dest = &dest->mgr;
    cinfo.image_width = data->width;
    cinfo.image_height = data->height;

    jpeg_create_decompress(&dinfo);
    dinfo.src = &src->mgr;

    args[0] = (VALUE)&dinfo;
    args[1] = (VALUE)&cinfo;
    args[2] = (VALUE)data;
    rb_protect(jpeg_each3, (VALUE)args, &state);

    jpeg_destroy_decompress(&dinfo);
    jpeg_destroy_compress(&cinfo);

    if (state) rb_jump_tag(state);
}

static void
jpeg_each(struct thumbdata *data)
{
    struct jpeg_src src;
    struct jpeg_dest dest;

    memset(&src, 0, sizeof(struct jpeg_src));
    src.io = data->io;
    src.buffer = rb_str_new(0, 0);
    src.mgr.init_source = init_source;
    src.mgr.fill_input_buffer = fill_input_buffer;
    src.mgr.skip_input_data = skip_input_data;
    src.mgr.resync_to_restart = jpeg_resync_to_restart;
    src.mgr.term_source = term_source;

    memset(&dest, 0, sizeof(struct jpeg_dest));
    dest.buffer = rb_str_new(0, BUF_LEN);    
    dest.mgr.next_output_byte = RSTRING_PTR(dest.buffer);
    dest.mgr.free_in_buffer = BUF_LEN;
    dest.mgr.init_destination = init_destination;
    dest.mgr.empty_output_buffer = empty_output_buffer;
    dest.mgr.term_destination = term_destination;

    jpeg_each2(data, &src, &dest);
}

/* png helper functions */

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

static VALUE
png_interlaced2(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    struct reader *reader = (struct reader *)args[0];
    struct writer *writer = (struct writer *)args[1];
    struct thumbdata *thumb = (struct thumbdata *)args[2];
    resize(reader, writer, thumb->precision);
}

static void
png_interlaced(png_structp rpng, struct reader *reader, struct writer *writer,
	       struct thumbdata *thumb)
{
    struct bitmap b;
    png_bytep *rows;
    char *data;
    long i;
    int state;
    VALUE args[3];

    b.rowlen = reader->width * reader->cmp;
    data = malloc(b.rowlen * reader->height);
    b.cur = data;

    rows = malloc(reader->height * sizeof(png_bytep));
    for (i=0; i<reader->height; i++)
	rows[i] = data + (i * b.rowlen);

    png_read_image(rpng, rows);

    reader->read = bitmap_read;
    reader->data = (void *)&b;
    args[0] = (VALUE)reader;
    args[1] = (VALUE)writer;
    args[2] = (VALUE)thumb;

    rb_protect(png_interlaced2, (VALUE)args, &state);

    free(rows);
    free(data);

    if (state) rb_jump_tag(state);
}

static VALUE
png_each2(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    png_structp write_ptr=(png_structp)args[0];
    png_infop write_i_ptr=(png_infop)args[1];
    png_structp read_ptr=(png_structp)args[2];
    png_infop read_i_ptr=(png_infop)args[3];
    struct thumbdata *thumb=(struct thumbdata *)args[4];
    struct reader reader;
    struct writer writer;
    png_byte ctype;

    png_read_info(read_ptr, read_i_ptr);
    png_normalize_input(read_ptr, read_i_ptr);
    ctype = png_get_color_type(read_ptr, read_i_ptr);
    
    reader.width = png_get_image_width(read_ptr, read_i_ptr);
    reader.height = png_get_image_height(read_ptr, read_i_ptr);
    fix_ratio(reader.width, reader.height, &thumb->width, &thumb->height);
    png_set_IHDR(write_ptr, write_i_ptr, thumb->width, thumb->height, 8,
		 ctype, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(write_ptr, write_i_ptr);

    switch (ctype) {
      case PNG_COLOR_TYPE_GRAY: reader.cmp = 1; break;
      case PNG_COLOR_TYPE_GRAY_ALPHA: reader.cmp = 2; break;
      case PNG_COLOR_TYPE_RGB: reader.cmp = 3; break;
      case PNG_COLOR_TYPE_RGB_ALPHA: reader.cmp = 4; break;
      default: rb_raise(rb_eRuntimeError, "png color type not supported.");
    }

    writer.width = thumb->width;
    writer.height = thumb->height;
    writer.write = png_write;
    writer.data = write_ptr;

    switch (png_get_interlace_type(read_ptr, read_i_ptr)) {
      case PNG_INTERLACE_NONE:
	reader.read = png_read;
	reader.data = read_ptr;
	resize(&reader, &writer, thumb->precision);
	break;
      case PNG_INTERLACE_ADAM7:
	png_interlaced(read_ptr, &reader, &writer, thumb);
	break;
      default: rb_raise(rb_eRuntimeError, "png interlace type not supported.");
    }

    png_write_end(write_ptr, write_i_ptr);

    return Qnil;
}

static void
png_each(struct thumbdata *thumb)
{
    int state;
    VALUE args[5];
    png_structp read_ptr, write_ptr;
    png_infop read_i_ptr, write_i_ptr;
    struct png_src src;

    write_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
					(png_error_ptr)png_error_fn,
					(png_error_ptr)png_warning_fn);
    write_i_ptr = png_create_info_struct(write_ptr);
    png_set_write_fn(write_ptr, NULL, png_write_data, png_flush_data);

    src.io = thumb->io;
    src.buffer = rb_str_new(0, 0);

    read_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, (png_voidp)NULL,
				      (png_error_ptr)png_error_fn,
				      (png_error_ptr)png_warning_fn);
    read_i_ptr = png_create_info_struct(read_ptr);
    png_set_read_fn(read_ptr, (void *)&src, png_read_data);
    png_set_sig_bytes(read_ptr, SIG_LEN);

    args[0] = (VALUE)write_ptr;
    args[1] = (VALUE)write_i_ptr;
    args[2] = (VALUE)read_ptr;
    args[3] = (VALUE)read_i_ptr;
    args[4] = (VALUE)thumb;
    rb_protect(png_each2, (VALUE)args, &state);

    png_destroy_read_struct(&read_ptr, &read_i_ptr, (png_info **)NULL);
    png_destroy_write_struct(&write_ptr, &write_i_ptr);

    if (state) rb_jump_tag(state);
}

/* Ruby allocator, deallocator, mark, and methods */

static void
deallocate(struct thumbdata *data)
{
    free(data);
}

static void
mark(struct thumbdata *data)
{
    if (!NIL_P(data->io))
	rb_gc_mark(data->io);
}

static VALUE
allocate(VALUE klass)
{
    struct thumbdata *data;
    return Data_Make_Struct(klass, struct thumbdata, mark, deallocate, data);
}

/*
 *  call-seq:
 *     Oil.new(io, width, height)              -> obj
 *     Oil.new(io, width, height, exact: true) -> obj
 *
 *  Creates a new resizer. +io+ must be an IO-like object that responds to
 *  #read(size, buffer) and #seek(size).
 *
 *  The resulting image will be scaled to fit in the box given by +width+ and
 *  +height+ while preserving the original aspect ratio.
 *
 *  The optional +exact+ argument instructs the resizer to use a slower and more
 *  precise resizing algorithm.
 */

static VALUE
initialize(int argc, VALUE *argv, VALUE self)
{
    VALUE io, rb_width, rb_height, options;
    long width, height;
    int precision;
    struct thumbdata *data;

    rb_scan_args(argc, argv, "31", &io, &rb_width, &rb_height, &options);

    width = FIX2INT(rb_width);
    height = FIX2INT(rb_height);

    if (width<1 || height<1) rb_raise(rb_eArgError, "Dimensions must be > 0");

    Data_Get_Struct(self, struct thumbdata, data);
    data->io = io;
    data->width = width;
    data->height = height;

    data->precision = RESIZE_FAST;
    if (TYPE(options) == T_HASH && RTEST(rb_hash_aref(options, sym_exact)))
	data->precision = RESIZE_EXACT;

    return self;
}

/*
 *  call-seq:
 *     oil.each(&block) -> self
 *
 *  Yields a series of binary strings that make up the resized image data.
 */

static VALUE
oil_each(VALUE self)
{
    struct thumbdata *thumb;
    VALUE string;
    unsigned char *cstr;

    Data_Get_Struct(self, struct thumbdata, thumb);
    string = rb_funcall(thumb->io, id_read, 1, INT2FIX(SIG_LEN));

    if (NIL_P(string) || RSTRING_LEN(string) != 2)
	rb_raise(rb_eRuntimeError, "Unable to read image signature.");

    cstr = RSTRING_PTR(string);

    if (!memcmp(cstr, jpeg_sig, SIG_LEN))
	jpeg_each(thumb);
    else if (!png_sig_cmp(cstr, 0, 2))
	png_each(thumb);
    else
	rb_raise(rb_eRuntimeError, "Unable to determine image type.");

    return self;
}

void
Init_oil()
{
    VALUE cOil = rb_define_class("Oil", rb_cObject);
    rb_define_alloc_func(cOil, allocate);
    rb_define_method(cOil, "initialize", initialize, -1);
    rb_define_method(cOil, "each", oil_each, 0);

    /* These are here for backward compatibility */
    rb_define_const(cOil, "JPEG", cOil);
    rb_define_const(cOil, "PNG", cOil);

    jpeg_std_error(&jerr);
    jerr.error_exit = error_exit;
    jerr.output_message = output_message;

    id_read = rb_intern("read");
    id_seek = rb_intern("seek");
    sym_exact = ID2SYM(rb_intern("exact"));

    jpeg_sig[0] = 0xFF;
    jpeg_sig[1] = 0xD8;

    jpeg_eof[0] = 0xFF;
    jpeg_eof[1] = JPEG_EOI;
}
