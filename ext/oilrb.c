#include <ruby.h>
#include "oil.h"

static ID id_read;
static VALUE sym_interpolation, sym_preserve_aspect_ratio, sym_point,
       sym_linear, sym_cubic;

struct thumbdata {
    struct image *reader;

    VALUE io;
    int initialized;
    int in_progress;

    enum image_type in_type;
    enum image_type out_type;

    VALUE interp;

    long out_width;
    long out_height;
};

static int
rb_read_fn(void *io, size_t len, unsigned char *buf, size_t *read_len)
{
    long strl;
    VALUE string;

    string = rb_funcall((VALUE)io, id_read, 1, INT2FIX(len));
    if (NIL_P(string)) {
	*read_len = 0;
	return 0;
    }

    if (TYPE(string) != T_STRING)
	rb_raise(rb_eTypeError, "IO returned something that is not a string.");

    strl = RSTRING_LEN(string);
    *read_len = (size_t)strl;
    if (*read_len > len)
	return -1;

    memcpy(buf, RSTRING_PTR(string), strl);
    return 0;
}

static void
free_reader(struct thumbdata *data)
{
    if (data->initialized)
	data->reader->free(data->reader);
}

static void
init_reader(struct thumbdata *data)
{
    int ret;

    data->initialized = 1;
    ret = oil_image_init(data->reader, rb_read_fn, (void *)data->io,
			 &data->in_type);
    data->out_type = data->in_type;
    if (ret >= 0)
	return;

    /* init failed */
    switch (ret) {
      case -1:
	rb_raise(rb_eRuntimeError, "IO Error.");
      case -2:
	rb_raise(rb_eRuntimeError, "Bad width.");
      case -3:
	rb_raise(rb_eRuntimeError, "Bad height.");
      case -4:
	rb_raise(rb_eRuntimeError, "Unsupported max color value.");
      case -5:
	rb_raise(rb_eRuntimeError, "Bad header.");
      default:
	rb_raise(rb_eRuntimeError, "Unknown Error.");
    }
}

static void
deallocate(struct thumbdata *data)
{
    free_reader(data);
    free(data->reader);
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
    VALUE self;

    self = Data_Make_Struct(klass, struct thumbdata, mark, deallocate, data);
    data->reader = malloc(sizeof(struct image));
    return self;
}

static void
fix_ratio(long sw, long sh, long *dw, long *dh)
{
    double x, y;

    x = *dw / (float)sw;
    y = *dh / (float)sh;

    if (x<y)
	*dh = sh * x;
    else
	*dw = sw * y;

    if (!*dh)
	*dh = 1;
    if (!*dw)
	*dw = 1;
}

static void
check_initialized(struct thumbdata *data)
{
    if (!data->initialized)
	rb_raise(rb_eRuntimeError, "Object is not initialized.");
}

static void
check_in_progress(struct thumbdata *data)
{
    if (data->in_progress)
	rb_raise(rb_eRuntimeError, "Transformation is already in progress.");
}

static void
set_interp(struct thumbdata *data, VALUE interp)
{
    if (interp != sym_point && interp != sym_linear && interp != sym_cubic)
	rb_raise(rb_eArgError, "Unknown scaling interpolator.");
    data->interp = interp;
}

static void
pre_scale(struct thumbdata *thumb)
{
    int inv_scale;
    int denom;

    if (thumb->in_type != JPEG)
	return;

    inv_scale = thumb->reader->width / thumb->out_width;

    if (inv_scale >= 8)
	denom = 8;
    else if (inv_scale >= 4)
	denom = 4;
    else if (inv_scale >= 2)
	denom = 2;
    else
    	return;

    jpeg_set_scale_denom(thumb->reader, denom);
}

static VALUE
oil_scale(int argc, VALUE *argv, VALUE self)
{
    VALUE rb_width, rb_height, options, interp, preserve_aspect;
    struct thumbdata *data;
    long w, h;
    int preserve_aspect_i;

    rb_scan_args(argc, argv, "21", &rb_width, &rb_height, &options);
    Data_Get_Struct(self, struct thumbdata, data);

    w = FIX2INT(rb_width);
    h = FIX2INT(rb_height);
    preserve_aspect_i = 1;

    if (TYPE(options) == T_HASH) {
	interp = rb_hash_aref(options, sym_interpolation);
	if (!NIL_P(interp))
	    set_interp(data, interp);

	preserve_aspect = rb_hash_aref(options, sym_preserve_aspect_ratio);
	if (!NIL_P(preserve_aspect) && !RTEST(preserve_aspect))
	    preserve_aspect_i = 0;
    }

    if (preserve_aspect_i)
	fix_ratio(data->reader->width, data->reader->height, &w, &h);

    if (w<1 || h<1)
	rb_raise(rb_eArgError, "Scale dimensions must be > 0");

    data->out_width = w;
    data->out_height = h;

    if (data->in_type == JPEG)
        jpeg_set_rgbx(data->reader);

    return self;
}

/*
 *  call-seq:
 *     Oil.new(io)                             -> obj
 *     Oil.new(io, width, height)              -> obj
 *
 *  Creates a new resizer. +io+ must be an IO-like object that responds to
 *  #read(size, buffer).
 *
 *  The resulting image will be scaled to fit in the box given by +width+ and
 *  +height+ while preserving the original aspect ratio.
 */

static VALUE
initialize(int argc, VALUE *argv, VALUE self)
{
    VALUE io, rb_width, rb_height, options;
    struct thumbdata *data;

    rb_scan_args(argc, argv, "13", &io, &rb_width, &rb_height, &options);
    Data_Get_Struct(self, struct thumbdata, data);

    data->io = io;
    data->interp = sym_point;

    free_reader(data);
    init_reader(data);

    if (argc > 1)
	oil_scale(argc - 1, argv + 1, self);

    return self;
}

static int
rb_write_fn(void *ctx, unsigned char *buf, size_t len)
{
    rb_yield(rb_str_new((char *)buf, len));
    return 0;
}

static VALUE
yield_resize(VALUE _writer)
{
    struct writer *writer;

    writer = (struct writer *)_writer;
    if (writer->write(writer))
	rb_raise(rb_eRuntimeError, "Writer Failed");
    return Qnil;
}

static void
init_equivalent_writer(struct writer *writer, enum image_type type,
		       struct image *src)
{
    switch (type) {
      case PPM:
        return ppm_writer_init(writer, rb_write_fn, 0, src);
      case JPEG:
        return jpeg_writer_init(writer, rb_write_fn, 0, src);
      case PNG:
        return png_writer_init(writer, rb_write_fn, 0, src);
    }
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
    struct image yscale, xscale;
    struct writer writer;
    struct thumbdata *thumb;
    int state;
    long w, h;

    Data_Get_Struct(self, struct thumbdata, thumb);
    check_initialized(thumb);
    check_in_progress(thumb);
    thumb->in_progress = 1;

    pre_scale(thumb);

    w = thumb->out_width;
    h = thumb->out_height;

    yscaler_init(&yscale, thumb->reader, h);
    xscaler_init(&xscale, &yscale, w);

    init_equivalent_writer(&writer, thumb->out_type, &xscale);
    rb_protect(yield_resize, (VALUE)&writer, &state);

    writer.free(&writer);
    xscale.free(&xscale);
    yscale.free(&yscale);

    thumb->in_progress = 0;
    if (state)
	rb_jump_tag(state);
    return self;
}

static VALUE
oil_width(VALUE self)
{
    struct thumbdata *thumb;
    Data_Get_Struct(self, struct thumbdata, thumb);
    check_initialized(thumb);
    return INT2FIX(thumb->reader->width);
}

static VALUE
oil_height(VALUE self)
{
    struct thumbdata *thumb;
    Data_Get_Struct(self, struct thumbdata, thumb);
    check_initialized(thumb);
    return INT2FIX(thumb->reader->height);
}

static VALUE
oil_out_width(VALUE self)
{
    struct thumbdata *thumb;
    Data_Get_Struct(self, struct thumbdata, thumb);
    check_initialized(thumb);
    return INT2FIX(thumb->out_width);
}

static VALUE
oil_out_height(VALUE self)
{
    struct thumbdata *thumb;
    Data_Get_Struct(self, struct thumbdata, thumb);
    check_initialized(thumb);
    return INT2FIX(thumb->out_height);
}

void
Init_oil()
{
    VALUE cOil = rb_define_class("Oil", rb_cObject);
    rb_define_alloc_func(cOil, allocate);
    rb_define_method(cOil, "initialize", initialize, -1);
    rb_define_method(cOil, "each", oil_each, 0);
    rb_define_method(cOil, "width", oil_width, 0);
    rb_define_method(cOil, "height", oil_height, 0);
    rb_define_method(cOil, "out_width", oil_out_width, 0);
    rb_define_method(cOil, "out_height", oil_out_height, 0);
    rb_define_method(cOil, "scale", oil_scale, -1);

    rb_define_const(cOil, "JPEG", cOil);
    rb_define_const(cOil, "PNG", cOil);

    id_read = rb_intern("read");
    sym_interpolation = ID2SYM(rb_intern("interpolation"));
    sym_preserve_aspect_ratio = ID2SYM(rb_intern("preserve_aspect_ratio"));
    sym_point = ID2SYM(rb_intern("point"));
    sym_linear = ID2SYM(rb_intern("linear"));
    sym_cubic = ID2SYM(rb_intern("cubic"));

    jpeg_appinit();
}
