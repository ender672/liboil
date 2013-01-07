#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "oil.h"

struct strip {
    struct image *src;
    int height;
    long pos;
    unsigned char **sl;
    unsigned char **virt;
};

typedef double (*intrp_fn_t)(double ty);

struct scaley {
    struct strip strip;
    long ypos;
    long tap_multiplier;
    double *coeffs;
    intrp_fn_t intrp;
};

struct scalex {
    struct image *src;
    unsigned char *buf;
    long base_taps;
    long tap_multiplier;
    double *coeffs;
    unsigned char **vals;
    intrp_fn_t intrp;
};

struct point {
    struct image *src;
    unsigned char *buf;
    long ypos;
    long src_ypos;
};

struct multi {
    struct image *ops;
    int num;
};

/* Map from a discreet dest coordinate to a continuous source coordinate.
 * The resulting coordinate can range from -0.5 to the maximum of the
 * destination image dimension.
 */
#define MAP(i, scale) (i + 0.5) / scale - 0.5

/* should do
 * pos = sl + BOUND(val, 0, max);
 * instead of
 * pos = BOUND(sl + val, sl, sl_max);
 */
#define BOUND(x, l, r) x <= l ? l : x >= r ? r : x

/* strip - A virtual band of scanlines */

static int
clamp(double d)
{
    if (d < 0)
	return 0;
    if (d > 255)
	return 255;
    return d;
}

static int
strip_move(struct strip *strip, long p)
{
    long target_pos, height, pos;
    int i, sl_pos;
    unsigned char **sl, **virt;

    pos = strip->pos;
    height = strip->height;
    target_pos = p + height / 2 + 1;
    sl = strip->sl;
    virt = strip->virt;

    while (pos < target_pos) {
	sl_pos = pos%height;

	for (i=1; i<height; i++)
	    virt[i - 1] = virt[i];

	if (pos < strip->src->height) {
	    if (strip->src->get_scanline(strip->src, sl[sl_pos]))
		return -1;
	    virt[height - 1] = sl[sl_pos];
	}

	pos++;
    }

    strip->pos = pos;
    return 0;
}

static void
strip_init(struct strip *strip, struct image *src, int height)
{
    long i;

    strip->src = src;
    strip->height = height;
    strip->sl = malloc(height * sizeof(unsigned char*));
    strip->virt = malloc(height * sizeof(unsigned char *));
    strip->pos = 0;

    for (i=0; i<strip->height; i++) {
	strip->sl[i] = malloc(src->width * src->cmp);
	strip->virt[i] = strip->sl[0];
    }
}

static void
strip_free(struct strip *strip)
{
    int i;

    for (i=0; i<strip->height; i++)
	free(strip->sl[i]);
    free(strip->sl);
    free(strip->virt);
}

/* Wrapper around multiple scalers */

static void
multi_free(struct image *im)
{
    int i;
    struct multi *m;
    struct image *op;

    m = (struct multi *)im->data;
    for (i=0; i<m->num; i++) {
	op = m->ops + i;
	if (op->free)
	    op->free(op);
    }
    free(m->ops);
    free(m);
}

static int
multi_get_scanline(struct image *i, unsigned char *sl_out)
{
    struct multi *m;
    struct image *op;

    m = (struct multi *)i->data;
    op = m->ops + m->num - 1;
    return (op->get_scanline(op, sl_out));
}

static void
multi_init(struct image *i, int num, struct image **ops) {
    struct multi *m;

    m = malloc(sizeof(struct multi));
    m->ops = malloc(num * sizeof(struct image));
    memset(m->ops, 0, num * sizeof(struct image));

    m->num = num;
    i->data = (void *)m;
    i->free = multi_free;
    i->get_scanline = multi_get_scanline;
    *ops = m->ops;
}

/* x scaling helpers */

static void
scalex_free(struct image *i)
{
    struct scalex *b;

    b = (struct scalex *)i->data;
    free(b->coeffs);
    free(b->vals);
    free(b->buf);
    free(i->data);
}

static int
scalex_get_scanline(struct image *im, unsigned char *sl_out)
{
    struct scalex *b;
    struct image *src;
    double scale, xsmp, tx, result;
    unsigned char *sl, *right;
    long xsmp_i, i, j, k, tap_mult, taps, win_pos;
    int cmp;

    b = (struct scalex *)im->data;
    src = b->src;
    sl = b->buf;
    cmp = im->cmp;
    if (src->get_scanline(src, sl))
	return -1;

    scale = im->width / (double)src->width;
    tap_mult = b->tap_multiplier;
    taps = tap_mult * b->base_taps;
    right = sl + (src->width - 1) * im->cmp;

    for (i=0; i<im->width; i++) {
	xsmp = MAP(i, scale);
	xsmp_i = xsmp < 0 ? -1 : (long)xsmp;
	tx = xsmp - xsmp_i;

	win_pos = -taps / 2 + 1;
	for (j=0; j<taps; j++) {
	    b->vals[j] = BOUND(sl + (xsmp_i + win_pos) * cmp, sl, right);
	    b->coeffs[j] = b->intrp((win_pos <= 0 ? -win_pos + tx : win_pos - tx) / tap_mult) / tap_mult;
	    win_pos++;
	}

	for (j=0; j<cmp; j++) {
	    result = 0;
	    for (k=0; k<taps; k++)
		result += b->coeffs[k] * b->vals[k][j];
	    sl_out[i * cmp + j] = clamp(result);
	}
    }
    return 0;
}

static void
scalex_init(struct image *i, struct image *src, long width, int taps,
	    intrp_fn_t intrp)
{
    struct scalex *b;
    long all_taps;

    i->width = width;
    i->height = src->height;
    i->cmp = src->cmp;
    i->free = scalex_free;
    i->get_scanline = scalex_get_scanline;

    b = malloc(sizeof(struct scalex));
    b->src = src;
    b->buf = malloc(src->width * i->cmp);
    b->base_taps = taps;
    b->intrp = intrp;

    b->tap_multiplier = src->width / width;
    if (b->tap_multiplier < 1)
	b->tap_multiplier = 1;
    all_taps = taps * b->tap_multiplier;
    b->coeffs = malloc(all_taps * sizeof(double));
    b->vals = malloc(all_taps * sizeof(unsigned char *));

    i->data = (void *)b;
}

/* y scaling helpers */

static int
scaley_get_scanline(struct image *im, unsigned char *sl_out)
{
    double ysmp, yscale, ty, result;
    long ysmp_i, i, j, taps, win_pos;
    unsigned char **sl;
    struct scaley *b;

    b = (struct scaley *)im->data;

    yscale = im->height / (double)b->strip.src->height;
    ysmp = MAP(b->ypos, yscale);
    ysmp_i = ysmp < 0 ? -1 : (long)ysmp;

    if (strip_move(&b->strip, ysmp_i))
	return -1;

    sl = b->strip.virt;
    taps = b->strip.height;
    ty = ysmp - ysmp_i;

    win_pos = -taps / 2 + 1;
    for (i=0; i<taps; i++) {
	b->coeffs[i] = b->intrp((win_pos<=0 ? -win_pos + ty : win_pos - ty) / b->tap_multiplier) / b->tap_multiplier;
	win_pos++;
    }

    for (i=0; i<im->width * im->cmp; i++) {
	result = 0;
	for (j=0; j<taps; j++)
	    result += b->coeffs[j] * sl[j][i];
	sl_out[i] = clamp(result);
    }
    b->ypos++;
    return 0;
}

static void
scaley_free(struct image *i)
{
    struct scaley *b;

    b = (struct scaley *)i->data;
    strip_free(&b->strip);
    free(b->coeffs);
    free(b);
}

static void
scaley_init(struct image *i, struct image *src, long height, int taps,
	    intrp_fn_t intrp)
{
    struct scaley *b;
    long all_taps;

    b = malloc(sizeof(struct scaley));
    b->ypos = 0;

    b->tap_multiplier = src->height / height;
    if (b->tap_multiplier < 1)
	b->tap_multiplier = 1;
    all_taps = taps * b->tap_multiplier;
    strip_init(&b->strip, src, all_taps);
    b->coeffs = malloc(all_taps * sizeof(double));
    b->intrp = intrp;

    i->width = src->width;
    i->height = height;
    i->cmp = src->cmp;
    i->free = scaley_free;
    i->get_scanline = scaley_get_scanline;
    i->data = (void *)b;
}

/* bicubic resizing */

static double
catrom(double x)
{
    if (x<1)
	return (9*x*x*x - 15*x*x + 6) / 6;
    return (-3*x*x*x + 15*x*x - 24*x + 12) / 6;
}

void
cubic_init(struct image *i, struct image *src, long width, long height) {
    struct image *ops;
    multi_init(i, 2, &ops);
    scaley_init(ops, src, height, 4, catrom);
    scalex_init(ops + 1, ops, width, 4, catrom);
    i->width = width;
    i->height = height;
    i->cmp = src->cmp;
}

/* bilinear resizing */

static double
bilinear_intrp(double ty)
{
    return 1 - ty;
}

void
linear_init(struct image *i, struct image *src, long width, long height) {
    struct image *ops;
    multi_init(i, 2, &ops);
    scalex_init(ops, src, width, 2, bilinear_intrp);
    scaley_init(ops + 1, ops, width, 2, bilinear_intrp);
    i->width = width;
    i->height = height;
    i->cmp = src->cmp;
}

/* point resizing */

void point_free(struct image *i)
{
    struct point *b;
    b = (struct point *)i->data;
    free(b->buf);
    free(b);
}

static int
point_get_scanline(struct image *im, unsigned char *sl_out)
{
    struct point *b;
    struct image *src;
    double scale;
    long smp, i, j;
    int cmp;

    b = (struct point *)im->data;
    src = b->src;
    cmp = im->cmp;

    scale = im->height / (double)src->height;
    smp = b->ypos / scale;

    while (b->src_ypos < smp) {
	if (src->get_scanline(src, b->buf))
	    return -1;
	b->src_ypos++;
    }

    scale = im->width / (double)src->width;

    for (i=0; i<im->width; i++) {
	smp = i / scale;
	for (j=0; j<cmp; j++)
	    sl_out[i * cmp + j] = b->buf[smp * cmp + j];
    }

    b->ypos++;
    return 0;
}

void point_init(struct image *i, struct image *src, long width, long height)
{
    struct point *b;

    i->width = width;
    i->height = height;
    i->cmp = src->cmp;
    i->free = point_free;
    i->get_scanline = point_get_scanline;

    b = malloc(sizeof(struct point));
    b->src = src;
    b->buf = malloc(src->width * src->cmp);
    b->ypos = 0;
    b->src_ypos = -1;

    i->data = (void *)b;
}

/* generic image reader */

#define SIG_SIZE 2

static int
sig_to_image_type(unsigned char *buf, enum image_type *type)
{
    if (buf[0] == 0x50 && buf[1] == 0x36)
	*type = PPM;
    else if (buf[0] == 0xFF && buf[1] == 0xD8)
	*type = JPEG;
    else if (buf[0] == 0x89 && buf[1] == 0x50)
	*type = PNG;
    else
	return -1;

    return 0;
}

static void
oil_image_free(struct image *i) { }

int
oil_image_init(struct image *i, read_fn_t read, void *ctx,
	       enum image_type *type)
{
    size_t read_len;
    unsigned char sig[2];

    i->free = oil_image_free;
    if (read(ctx, SIG_SIZE, sig, &read_len) || read_len != SIG_SIZE)
	return -1; // IO error

    if (sig_to_image_type(sig, type))
	return -6; // Unknown signature

    switch (*type) {
      case PPM:
	return ppm_init(i, read, ctx, SIG_SIZE);
      case JPEG:
	return jpeg_init(i, read, ctx, SIG_SIZE);
      case PNG:
	return png_init(i, read, ctx, SIG_SIZE);
    }

    return -3; // Signature detection is broken
}
