#include <ruby.h>
#include <jpeglib.h>
#include <png.h>

#define BUF_LEN 8192

/* Map from discreet dest coordinate to scaled continuous source coordinate */
#define MAP(i, scale) (i + 0.5) / scale - 0.5;

static ID id_read, id_seek;

struct thumbdata {
    VALUE io;
    unsigned int width;
    unsigned int height;
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

struct interpolation {
    long sw;
    long sh;
    long dw;
    long dh;
    int cmp;
    long source_line;
    void (*read)(char *, void *);
    void (*write)(char *, void *);
    void *read_data;
    void *write_data;
};

struct bitmap {
    long rowlen;
    char *cur;
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
static void init_source(j_decompress_ptr cinfo){};

static boolean
fill_input_buffer(j_decompress_ptr cinfo)
{
    struct jpeg_src *src = (struct jpeg_src *)cinfo->src;
    VALUE ret, string = src->buffer;
    long len;
    char *buf;

    ret = rb_funcall(src->io, id_read, 2, INT2FIX(BUF_LEN), string);
    len = RSTRING_LEN(string);

    if (!len || NIL_P(ret)) {
	rb_str_resize(string, 2);
        buf = RSTRING_PTR(string);
        buf[0] = 0xFF;
	buf[1] = JPEG_EOI;
	len = 2;
    }
    src->mgr.next_input_byte = (JOCTET *)RSTRING_PTR(string);
    src->mgr.bytes_in_buffer = (size_t)len;

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

/* Bilinear Interpolation */

static void
bilinear_get_scanlines(struct interpolation *bi, char **sl1, char **sl2,
		       int line)
{
    int i, j, n, len=bi->cmp * bi->sw - 1;
    char *tmp;

    n = line + 1 - bi->source_line;
    if (line < bi->sh - 1) n++;
    bi->source_line += n;

    for (i=0; i<n; i++) {
	tmp = *sl1;
	*sl1 = *sl2;
	*sl2 = tmp;
	bi->read(*sl2, bi->read_data);

	for (j=0; j<bi->cmp; j++)
	    sl2[0][len + bi->cmp - j] = sl2[0][len - j];
    }

    if (line >= bi->sh - 1) *sl1 = *sl2;
}

static void
bilinear3(char *sl1, char *sl2, char *sl_out, double scale, size_t dw, int cmp,
	  double ty)
{
    double xsmp, tx, p1, p2, p3, p4;
    unsigned char *c1, *c2, *c3, *c4;
    size_t xsmp_i, i, j;

    for (i = 0; i < dw; i++) {
	xsmp = MAP(i, scale);
	if (xsmp < 0) xsmp = 0;
	xsmp_i = (int)xsmp;
	tx = xsmp - xsmp_i;

	p4 =  tx * ty;
	p3 = (1 - tx) * ty;
	p2 =  tx - p4;
	p1 = (1 - tx) - p3;

	c1 = sl1 + xsmp_i * cmp;
	c2 = c1 + cmp;
	c3 = sl2 + xsmp_i * cmp;
	c4 = c3 + cmp;

	for (j = 0; j < cmp; j++)
	    *sl_out++ = p1 * c1[j] + p2 * c2[j] + p3 * c3[j] + p4 * c4[j];
    }
}

static VALUE
bilinear2(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    struct interpolation *bi=(struct interpolation *)args[0];
    char *sl1=(char *)args[1], *sl2=(char *)args[2], *sl_out=(char *)args[3];
    size_t ysmp_i, i;
    double ysmp, xscale, yscale, ty;

    yscale = bi->dh / (float)bi->sh;
    xscale = bi->dw / (float)bi->sw;
    bi->source_line = 0;

    for (i = 0; i<bi->dh; i++) {
	ysmp = MAP(i, yscale);
	if (ysmp < 0) ysmp = 0;
	ysmp_i = (int)ysmp;
	ty = ysmp - ysmp_i;

	bilinear_get_scanlines(bi, &sl1, &sl2, ysmp_i);
	bilinear3(sl1, sl2, sl_out, xscale, bi->dw, bi->cmp, ty);
	bi->write(sl_out, bi->write_data);
    }

    return Qnil;
}

static void
bilinear(struct interpolation *bi)
{
    VALUE args[4];
    int state, in_len;
    char *sl1, *sl2, *sl_out;

    in_len = (bi->sw + 1) * bi->cmp;
    sl1 = malloc(in_len);
    sl2 = malloc(in_len);
    sl_out = malloc(bi->dw * bi->cmp);

    args[0] = (VALUE)bi;
    args[1] = (VALUE)sl1;
    args[2] = (VALUE)sl2;
    args[3] = (VALUE)sl_out;
    rb_protect(bilinear2, (VALUE)args, &state);

    free(sl1);
    free(sl2);
    free(sl_out);
 
    if (state) rb_jump_tag(state);
}

/* helper functions */

static void
fix_ratio(unsigned int sw, unsigned int sh, unsigned int *dw, unsigned int *dh)
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
    struct interpolation intrp;

    jpeg_read_header(dinfo, TRUE);
    jpeg_calc_output_dimensions(dinfo);
    jpeg_set_output_header(cinfo, dinfo);
    fix_ratio(dinfo->output_width, dinfo->output_height, &cinfo->image_width,
	      &cinfo->image_height);
    jpeg_pre_scale(cinfo, dinfo);

    jpeg_start_compress(cinfo, TRUE);
    jpeg_start_decompress(dinfo);

    intrp.sw = dinfo->output_width;
    intrp.sh = dinfo->output_height;
    intrp.dw = cinfo->image_width;
    intrp.dh = cinfo->image_height;
    intrp.cmp = dinfo->output_components;
    intrp.read = jpeg_read;
    intrp.write = jpeg_write;
    intrp.read_data = dinfo;
    intrp.write_data = cinfo;

    bilinear(&intrp);

    jpeg_abort_decompress(dinfo);
    jpeg_finish_compress(cinfo);

    return Qnil;
}

static void
jpeg_each2(struct thumbdata *data, struct jpeg_src *src, struct jpeg_dest *dest)
{
    int state;
    VALUE args[2];
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
    rb_protect(jpeg_each3, (VALUE)args, &state);

    jpeg_destroy_decompress(&dinfo);
    jpeg_destroy_compress(&cinfo);

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
 *     Class.new(io, width, height) -> obj
 *
 *  Creates a new resizer. +io_in+ must be an IO-like object that responds to
 *  #read(size, buffer) and #seek(size).
 * 
 *  The resulting image will be scaled to fit in the box given by +width+ and
 *  +height+ while preserving the original aspect ratio.
 */

static VALUE
initialize(VALUE self, VALUE io, VALUE rb_width, VALUE rb_height)
{
    int width=FIX2INT(rb_width), height=FIX2INT(rb_height);
    struct thumbdata *data;
    
    if (width<1 || height<1) rb_raise(rb_eArgError, "Dimensions must be > 0");

    Data_Get_Struct(self, struct thumbdata, data);
    data->io = io;
    data->width = width;
    data->height = height;

    return self;
}

/*
 *  call-seq:
 *     jpeg.each(&block) -> self
 *
 *  Yields a series of binary strings that make up the resized JPEG image.
 */

static VALUE
jpeg_each(VALUE self)
{
    struct jpeg_src src;
    struct jpeg_dest dest;
    struct thumbdata *data;
    Data_Get_Struct(self, struct thumbdata, data);

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

    return self;
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

static VALUE
png_interlaced2(VALUE arg)
{
    bilinear((struct interpolation *)arg);
}

static void
png_interlaced(png_structp rpng, struct interpolation *intrp)
{
    struct bitmap b;
    png_bytep *rows;
    char *data;
    int i, state;

    b.rowlen = intrp->sw * intrp->cmp;
    data = malloc(b.rowlen * intrp->sh);
    b.cur = data;

    rows = malloc(intrp->sh * sizeof(png_bytep));
    for (i=0; i<intrp->sh; i++)
	rows[i] = data + (i * b.rowlen);

    png_read_image(rpng, rows);

    intrp->read = bitmap_read;
    intrp->read_data = (void *)&b;

    rb_protect(png_interlaced2, (VALUE)intrp, &state);

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
    struct interpolation intrp;
    png_byte ctype;

    png_read_info(read_ptr, read_i_ptr);
    png_normalize_input(read_ptr, read_i_ptr);
    ctype = png_get_color_type(read_ptr, read_i_ptr);
    
    intrp.sw = png_get_image_width(read_ptr, read_i_ptr);
    intrp.sh = png_get_image_height(read_ptr, read_i_ptr);
    fix_ratio(intrp.sw, intrp.sh, &thumb->width, &thumb->height);
    png_set_IHDR(write_ptr, write_i_ptr, thumb->width, thumb->height, 8,
		 ctype, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
		 PNG_FILTER_TYPE_DEFAULT);
    png_write_info(write_ptr, write_i_ptr);

    switch (ctype) {
      case PNG_COLOR_TYPE_GRAY: intrp.cmp = 1; break;
      case PNG_COLOR_TYPE_GRAY_ALPHA: intrp.cmp = 2; break;
      case PNG_COLOR_TYPE_RGB: intrp.cmp = 3; break;
      case PNG_COLOR_TYPE_RGB_ALPHA: intrp.cmp = 4; break;
      default: rb_raise(rb_eRuntimeError, "png color type not supported.");
    }

    intrp.dw = thumb->width;
    intrp.dh = thumb->height;
    intrp.write = png_write;
    intrp.write_data = write_ptr;
    
    switch (png_get_interlace_type(read_ptr, read_i_ptr)) {
      case PNG_INTERLACE_NONE:
	intrp.read = png_read;
	intrp.read_data = read_ptr;
	bilinear(&intrp);
	break;
      case PNG_INTERLACE_ADAM7:
	png_interlaced(read_ptr, &intrp);
	break;
      default: rb_raise(rb_eRuntimeError, "png interlace type not supported.");
    }

    png_write_end(write_ptr, write_i_ptr);

    return Qnil;
}

/*
 *  call-seq:
 *     png.each(&block) -> self
 *
 *  Yields a series of binary strings that make up the resized PNG image.
 */

static VALUE
png_each(VALUE self)
{
    int state;
    VALUE args[5];
    png_structp read_ptr, write_ptr;
    png_infop read_i_ptr, write_i_ptr;
    struct png_src src;
    struct thumbdata *thumb;
    Data_Get_Struct(self, struct thumbdata, thumb);

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

    args[0] = (VALUE)write_ptr;
    args[1] = (VALUE)write_i_ptr;
    args[2] = (VALUE)read_ptr;
    args[3] = (VALUE)read_i_ptr;
    args[4] = (VALUE)thumb;
    rb_protect(png_each2, (VALUE)args, &state);

    png_destroy_read_struct(&read_ptr, &read_i_ptr, (png_info **)NULL);
    png_destroy_write_struct(&write_ptr, &write_i_ptr);

    if (state) rb_jump_tag(state);
    return self;
}

void
Init_oil(void)
{
    VALUE mOil = rb_define_module("Oil");

    VALUE cJPEG = rb_define_class_under(mOil, "JPEG", rb_cObject);
    rb_define_alloc_func(cJPEG, allocate);
    rb_define_method(cJPEG, "initialize", initialize, 3);
    rb_define_method(cJPEG, "each", jpeg_each, 0);
    
    VALUE cPNG = rb_define_class_under(mOil, "PNG", rb_cObject);
    rb_define_alloc_func(cPNG, allocate);
    rb_define_method(cPNG, "initialize", initialize, 3);
    rb_define_method(cPNG, "each", png_each, 0);

    jpeg_std_error(&jerr);
    jerr.error_exit = error_exit;
    jerr.output_message = output_message;

    id_read = rb_intern("read");
    id_seek = rb_intern("seek");
}
