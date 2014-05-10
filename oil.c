#include "oil.h"
#include <stdlib.h>
#include <string.h>

/* yscaler */

struct yscaler {
    struct strip st;
    struct image *src;
};

static int
yscaler_get_src_scanline(struct yscaler *ys)
{
    struct image *src;
    unsigned char *buf;

    src = ys->src;
    while ((buf = strip_next_inbuf(&ys->st)))
        if (src->get_scanline(src, buf))
            return -1;
    return 0;
}

static int
yscaler_get_scanline(struct image *im, unsigned char *sl_out)
{
    struct yscaler *ys;
    ys = (struct yscaler *)im->data;
    if (yscaler_get_src_scanline(ys))
        return -1;
    strip_scale(&ys->st, sl_out);
    return 0;
}

static void
yscaler_free(struct image *im)
{
    struct yscaler *ys;
    ys = (struct yscaler *)im->data;
    strip_free(&ys->st);
    free(ys);
}

void
yscaler_init(struct image *im, struct image *src, long height)
{
    struct yscaler *ys;

    ys = malloc(sizeof(struct yscaler));
    ys->src = src;
    strip_init(&ys->st, src->height, height, src->width, src->fmt);

    im->fmt = src->fmt;
    im->width = src->width;
    im->height = height;
    im->free = yscaler_free;
    im->get_scanline = yscaler_get_scanline;
    im->data = (void *)ys;
}

/* xscaler */

struct xscaler {
    struct padded_sl psl;
    struct image *src;
};

static void
xscaler_free(struct image *im)
{
    struct xscaler *xs;
    xs = (struct xscaler *)im->data;
    padded_sl_free(&xs->psl);
    free(xs);
}

static int
xscaler_get_src_scanline(struct xscaler *xs)
{
    struct image *src;
    src = xs->src;
    return src->get_scanline(src, xs->psl.buf);
}

static int
xscaler_get_scanline(struct image *im, unsigned char *sl_out)
{
    struct xscaler *xs;
    xs = (struct xscaler *)im->data;
    if (xscaler_get_src_scanline(xs))
        return -1;
    padded_sl_scale(&xs->psl, sl_out);
    return 0;
}

void
xscaler_init(struct image *im, struct image *src, long width)
{
    struct xscaler *xs;

    xs = malloc(sizeof(struct xscaler));

    xs->src = src;
    padded_sl_init(&xs->psl, src->width, width, src->fmt);

    im->fmt = src->fmt;
    im->height = src->height;
    im->width = width;
    im->free = xscaler_free;
    im->get_scanline = xscaler_get_scanline;
    im->data = (void *)xs;
}

/* random image generator */
static int
random_get_scanline(struct image *im, unsigned char *sl_out)
{
    memset(sl_out, rand(), im->width * sample_size(im->fmt));
    return 0;
}

void random_free(struct image *im) { }

void
random_init(struct image *im, enum sample_fmt fmt, long width, long height)
{
    im->fmt = fmt;
    im->height = height;
    im->width = width;
    im->free = random_free;
    im->get_scanline = random_get_scanline;
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

int
ext_to_image_type(char *ext, enum image_type *type)
{
    ext++;

    if (!strcmp(ext, "ppm"))
        *type = PPM;
    else if (!strcmp(ext, "jpg") || !strcmp(ext, "jpeg"))
        *type = JPEG;
    else if (!strcmp(ext, "png"))
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
	return -1; /* IO error */

    if (sig_to_image_type(sig, type))
	return -6; /* Unknown signature */

    switch (*type) {
      case PPM:
	return ppm_init(i, read, ctx, SIG_SIZE);
      case JPEG:
	return jpeg_init(i, read, ctx, SIG_SIZE);
      case PNG:
	return png_init(i, read, ctx, SIG_SIZE);
    }

    return -3; /* Signature detection is broken */
}
