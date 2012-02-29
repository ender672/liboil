#include <ruby.h>
#include <jpeglib.h>

#define BUF_LEN 8192

static ID id_read, id_seek;

struct thumbdata {
    VALUE io;
    long width;
    long height;
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

/* Helper functions */

static void
advance_scanlines(j_decompress_ptr dinfo, JSAMPROW *sl1, JSAMPROW *sl2, int n)
{
    int i;
    JSAMPROW tmp;
    for (i = 0; i < n; i++) {
	tmp = *sl1;
	*sl1 = *sl2;
	*sl2 = tmp;
	jpeg_read_scanlines(dinfo, sl2, 1);
    }
}

static void
bilinear3(j_compress_ptr cinfo, char *sl1, char *sl2, char *sl_out,
	  double width_ratio_inv, double ty)
{
    double sample_x, tx, _tx, p00, p10, p01, p11;
    unsigned char *c00, *c10, *c01, *c11;
    size_t sample_x_i, i, j, cmp = cinfo->input_components;
    
    for (i = 0; i < cinfo->image_width; i++) {
	sample_x = i * width_ratio_inv;
	sample_x_i = (int)sample_x;

	tx = sample_x - sample_x_i;
	_tx = 1 - tx;

	p11 =  tx * ty;
	p01 = _tx * ty;
	p10 =  tx - p11;
	p00 = _tx - p01;

	c00 = sl1 + sample_x_i * cmp;
	c10 = c00 + cmp;
	c01 = sl2 + sample_x_i * cmp;
	c11 = c01 + cmp;

	for (j = 0; j < cmp; j++)
	    *sl_out++ = p00 * c00[j] + p10 * c10[j] + p01 * c01[j] +
			 p11 * c11[j];
    }
}

static VALUE
bilinear2(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    j_compress_ptr cinfo = (j_compress_ptr)args[0];
    j_decompress_ptr dinfo = (j_decompress_ptr)args[1];
    JSAMPROW sl1 = (JSAMPROW)args[2];
    JSAMPROW sl2 = (JSAMPROW)args[3];
    JSAMPROW sl_out = (JSAMPROW)args[4];
    size_t i, smpy_i, smpy_last=0, sl_len;
    double smpy, ty, inv_scale_y, inv_scale_x;

    sl_len = (dinfo->output_width + 1) * dinfo->output_components;
    inv_scale_x = dinfo->output_width / (float)cinfo->image_width;
    inv_scale_y = dinfo->output_height / (float)cinfo->image_height;
    advance_scanlines(dinfo, &sl1, &sl2, 1);

    for (i = 0; i < cinfo->image_height; i++) {
	smpy = i * inv_scale_y;
	smpy_i = (int)smpy;
	ty = smpy - smpy_i;
	
	advance_scanlines(dinfo, &sl1, &sl2, smpy_i - smpy_last);
	smpy_last = smpy_i;

	sl1[sl_len - 1] = sl1[sl_len - 2];
	sl2[sl_len - 1] = sl2[sl_len - 2];
	bilinear3(cinfo, smpy_i ? sl1 : sl2, sl2, sl_out, inv_scale_x, ty);
	jpeg_write_scanlines(cinfo, &sl_out, 1);
    }

    return Qnil;
}

static void
bilinear(j_compress_ptr cinfo, j_decompress_ptr dinfo)
{
    int state;
    VALUE args[5];
    size_t sl_len;
    JSAMPROW sl1, sl2, sl_out;
 
    sl_len = (dinfo->output_width + 1) * dinfo->output_components;    
    sl1 = (JSAMPROW)malloc(sl_len);
    sl2 = (JSAMPROW)malloc(sl_len);
    sl_out = (JSAMPROW)malloc(cinfo->image_width * cinfo->input_components);

    args[0] = (VALUE)cinfo;
    args[1] = (VALUE)dinfo;
    args[2] = (VALUE)sl1;
    args[3] = (VALUE)sl2;
    args[4] = (VALUE)sl_out;
    rb_protect(bilinear2, (VALUE)args, &state);
    
    free(sl1);
    free(sl2);
    free(sl_out);
    
    if (state) rb_jump_tag(state);
}

static void
fix_aspect_ratio(j_compress_ptr cinfo, j_decompress_ptr dinfo)
{
    double x, y;

    x = cinfo->image_width / (float)dinfo->output_width;
    y = cinfo->image_height / (float)dinfo->output_height;
    
    if (x < y) cinfo->image_height = dinfo->output_height * x;
    else cinfo->image_width = dinfo->output_width * y;
}

static void
set_output_header(j_compress_ptr cinfo, j_decompress_ptr dinfo)
{
    cinfo->input_components = dinfo->output_components;
    cinfo->in_color_space = dinfo->out_color_space;
    jpeg_set_defaults(cinfo);
}

static void
pre_scale(j_compress_ptr cinfo, j_decompress_ptr dinfo)
{
    int inv_scale = dinfo->output_width / cinfo->image_width;

    if (inv_scale >= 8) dinfo->scale_denom = 8;
    else if (inv_scale >= 4) dinfo->scale_denom = 4;
    else if (inv_scale >= 2) dinfo->scale_denom = 2;
    jpeg_calc_output_dimensions(dinfo);
}

static VALUE
each3(VALUE _args)
{
    VALUE *args = (VALUE *)_args;
    j_decompress_ptr dinfo = (j_decompress_ptr)args[0];
    j_compress_ptr cinfo = (j_compress_ptr)args[1];

    jpeg_read_header(dinfo, TRUE);
    jpeg_calc_output_dimensions(dinfo);
    set_output_header(cinfo, dinfo);
    fix_aspect_ratio(cinfo, dinfo);
    pre_scale(cinfo, dinfo);

    jpeg_start_compress(cinfo, TRUE);
    jpeg_start_decompress(dinfo);

    bilinear(cinfo, dinfo);

    jpeg_abort_decompress(dinfo);
    jpeg_finish_compress(cinfo);

    return Qnil;
}

static void
each2(struct thumbdata *data, struct jpeg_src *src, struct jpeg_dest *dest)
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
    rb_protect(each3, (VALUE)args, &state);

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
 *     JPEGThumb.new(io, width, height) -> jpeg_thumb
 *
 *  Creates a new JPEG thumbnailer. +io_in+ must be an IO-like object that
 *  responds to #read(size).
 * 
 *  The resulting image will be scaled to fit in the box given by +width+ and
 *  +height+ while preserving the aspect ratio.
 * 
 *  If both box dimensions are larger than the source image, then the image will
 *  be unchanged. The thumbnailer will not enlarge images.
 */

static VALUE
initialize(VALUE self, VALUE io, VALUE width, VALUE height)
{
    struct thumbdata *data;

    Data_Get_Struct(self, struct thumbdata, data);
    data->io = io;
    data->width = FIX2INT(width);
    data->height = FIX2INT(height);

    return self;
}

/*
 *  call-seq:
 *     resized_jpeg.each(&block) -> self
 *
 *  Yields a series of binary strings that make up the resized JPEG image.
 */

static VALUE
each(VALUE self)
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

    each2(data, &src, &dest);

    return self;
}

void
Init_oil()
{
    VALUE mOil = rb_define_module("Oil");
    VALUE cJPEG = rb_define_class_under(mOil, "JPEG", rb_cObject);
    rb_define_alloc_func(cJPEG, allocate);
    rb_define_method(cJPEG, "initialize", initialize, 3);
    rb_define_method(cJPEG, "each", each, 0);

    jpeg_std_error(&jerr);
    jerr.error_exit = error_exit;
    jerr.output_message = output_message;

    id_read = rb_intern("read");
    id_seek = rb_intern("seek");
}

