#include <ruby.h>
#include <jpeglib.h>

#define BUF_LEN 4096

static ID id_read;

struct thumbdata {
    VALUE io;
    long width;
    long height;
};

static struct jpeg_error_mgr jerr;

struct jpeg_dest {
    struct jpeg_destination_mgr pub;
    VALUE buffer;
};

struct jpeg_src {
    struct jpeg_decompress_struct dinfo;
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
    size_t nbytes = 0;
    JOCTET *buffer = NULL;
    VALUE string;
    struct jpeg_src *src = (struct jpeg_src *)cinfo;

    string = src->buffer;
    
    rb_funcall(src->io, id_read, 2, INT2FIX(BUF_LEN), string);

    if (!NIL_P(string)) {
	StringValue(string);
	nbytes = (size_t)RSTRING_LEN(string);
	buffer = (JOCTET *)RSTRING_PTR(string);
    }

    src->mgr.next_input_byte = buffer;
    src->mgr.bytes_in_buffer = nbytes;    

    return TRUE;
}

static void
skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
    struct jpeg_source_mgr *src = cinfo->src;

    if (num_bytes > 0) {
	while (num_bytes > (long) src->bytes_in_buffer) {
	    num_bytes -= (long) src->bytes_in_buffer;
	    fill_input_buffer(cinfo);
	}
	src->next_input_byte += (size_t) num_bytes;
	src->bytes_in_buffer -= (size_t) num_bytes;
    }
}

/* jpeglib destination callbacks */

static void
reset_buffer(struct jpeg_dest *dest)
{
    dest->pub.next_output_byte = RSTRING_PTR(dest->buffer);
    dest->pub.free_in_buffer = BUF_LEN;
}

static void
init_destination(j_compress_ptr cinfo)
{
    struct jpeg_dest *dest = (struct jpeg_dest *)cinfo->dest;
    dest->buffer = rb_str_new(0, BUF_LEN);
    reset_buffer(dest);
}

static boolean
empty_output_buffer(j_compress_ptr cinfo)
{
    struct jpeg_dest *dest = (struct jpeg_dest *)cinfo->dest;
    rb_yield(dest->buffer);
    reset_buffer(dest);
    return TRUE;
}

static void
term_destination(j_compress_ptr cinfo)
{
    struct jpeg_dest *dest = (struct jpeg_dest *)cinfo->dest;
    size_t len = BUF_LEN - dest->pub.free_in_buffer;

    if (len) {
	rb_str_resize(dest->buffer, len);
	rb_yield(dest->buffer);
    }
}

/* Helper functions */

static void
advance_scanlines(struct jpeg_decompress_struct *dinfo, JSAMPROW *sl1,
		  JSAMPROW *sl2, int n)
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
bilinear2(struct jpeg_compress_struct *cinfo, char *sl1, char *sl2,
	  char *sl_out, double width_ratio_inv, double ty)
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

static void
bilinear(struct jpeg_compress_struct *cinfo,
	 struct jpeg_decompress_struct *dinfo)
{
    size_t i, smpy_i, smpy_last=0, sl_len;
    double smpy, ty, inv_scale_y, inv_scale_x;
    JSAMPROW sl1, sl2, sl_out;
    
    inv_scale_x = dinfo->output_width / (float)cinfo->image_width;
    inv_scale_y = dinfo->output_height / (float)cinfo->image_height;
    sl_len = (dinfo->output_width + 1) * dinfo->output_components;    

    sl1 = (JSAMPROW)malloc(sl_len);
    sl2 = (JSAMPROW)malloc(sl_len);
    sl_out = (JSAMPROW)malloc(cinfo->image_width * cinfo->input_components);
    advance_scanlines(dinfo, &sl1, &sl2, 1);
    
    for (i = 0; i < cinfo->image_height; i++) {
	smpy = i * inv_scale_y;
	smpy_i = (int)smpy;
	ty = smpy - smpy_i;
	
	advance_scanlines(dinfo, &sl1, &sl2, smpy_i - smpy_last);
	smpy_last = smpy_i;

	sl1[sl_len - 1] = sl1[sl_len - 2];
	sl2[sl_len - 1] = sl2[sl_len - 2];
	bilinear2(cinfo, smpy_i ? sl1 : sl2, sl2, sl_out, inv_scale_x, ty);
	jpeg_write_scanlines(cinfo, &sl_out, 1);
    }

    free(sl1);
    free(sl2);
    free(sl_out);
}

static double
fit_scale(VALUE self, struct jpeg_decompress_struct *dinfo)
{
    double x, y;
    struct thumbdata *data;
    Data_Get_Struct(self, struct thumbdata, data);

    x = data->width / (float)dinfo->output_width;
    y = data->height / (float)dinfo->output_height;

    return x < y ? x : y;
}

static void
set_header(VALUE self, struct jpeg_compress_struct *cinfo,
	   struct jpeg_decompress_struct *dinfo)
{
    double scale = fit_scale(self, dinfo);

    cinfo->image_width = dinfo->output_width * scale;
    cinfo->image_height = dinfo->output_height * scale;
    cinfo->input_components = dinfo->output_components;
    cinfo->in_color_space = dinfo->out_color_space;

    jpeg_set_defaults(cinfo);
    jpeg_set_quality(cinfo, 90, TRUE);
}

static void
pre_scale(VALUE self, struct jpeg_decompress_struct *dinfo)
{
    int inv_scale_i = 1 / fit_scale(self, dinfo);

    if (inv_scale_i >= 8) dinfo->scale_denom = 8;
    else if (inv_scale_i >= 4) dinfo->scale_denom = 4;
    else if (inv_scale_i >= 2) dinfo->scale_denom = 2;
    jpeg_calc_output_dimensions(dinfo);
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

static void
each3(VALUE self, struct jpeg_compress_struct *cinfo,
      struct jpeg_decompress_struct *dinfo)
{
    jpeg_read_header(dinfo, TRUE);
    jpeg_calc_output_dimensions(dinfo);
    set_header(self, cinfo, dinfo);
    pre_scale(self, dinfo);
    
    jpeg_start_compress(cinfo, TRUE);
    jpeg_start_decompress(dinfo);

    bilinear(cinfo, dinfo);

    jpeg_abort_decompress(dinfo);
    jpeg_finish_compress(cinfo);
}

static void
each2(VALUE self, struct jpeg_compress_struct *cinfo)
{
    struct jpeg_src src;
    struct thumbdata *data;
    Data_Get_Struct(self, struct thumbdata, data);
    
    memset(&src, 0, sizeof(struct jpeg_src));
    src.dinfo.err = &jerr;
    jpeg_create_decompress(&src.dinfo);

    src.dinfo.src = &src.mgr;
    src.mgr.init_source = init_source;
    src.mgr.fill_input_buffer = fill_input_buffer;
    src.mgr.skip_input_data = skip_input_data;
    src.mgr.resync_to_restart = jpeg_resync_to_restart;
    src.mgr.term_source = term_source;
    src.io = data->io;
    src.buffer = rb_str_new(0, BUF_LEN);    

    each3(self, cinfo, &src.dinfo);
    
    jpeg_destroy_decompress(&src.dinfo);
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
    struct jpeg_dest dest;
    struct jpeg_compress_struct cinfo;

    dest.pub.init_destination = init_destination;
    dest.pub.empty_output_buffer = empty_output_buffer;
    dest.pub.term_destination = term_destination;

    cinfo.err = &jerr;
    jpeg_create_compress(&cinfo);
    cinfo.dest = (struct jpeg_destination_mgr *)&dest;
    
    each2(self, &cinfo);
    
    jpeg_destroy_compress(&cinfo);
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
}
