#include "oil.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cmd_opt {
    int box;
    int rgbx;
    int prescale;
    int xy;
    int png;
    long width;
    long height;
    char *infile;
    char *outfile;
};

static int
stream_read_fn(void *ctx, size_t len, unsigned char *buf, size_t *read_len)
{
    *read_len = fread(buf, 1, len, (FILE*)ctx);
    if (!*read_len)
        return -1;

    return 0;
}

static int
stream_write_fn(void *ctx, unsigned char *buf, size_t len)
{
    size_t write_len;

    write_len = fwrite(buf, 1, len, (FILE*)ctx);

    if (write_len == len)
        return 0;

    return -1;
}

static void
fix_ratio(long sw, long sh, long *dw, long *dh)
{
    double x, y;

    x = *dw / (double)sw;
    y = *dh / (double)sh;

    if (x && (!y || x<y))
        *dh = (sh * x) + 0.5;
    else
        *dw = (sw * y) + 0.5;

    if (!*dh)
        *dh = 1;
    if (!*dw)
        *dw = 1;
}

static int
calc_pre_scale(long in_width, long out_width)
{
    int inv_scale;

    inv_scale = in_width / out_width;

    if (inv_scale >= 8)
        return 8;
    else if (inv_scale >= 4)
        return 4;
    else if (inv_scale >= 2)
        return 2;
    return 0;
}

void
prescale(struct image *im, long out_width, long out_height)
{
    int xscale, yscale, scale;

    xscale = calc_pre_scale(im->width, out_width);
    yscale = calc_pre_scale(im->height, out_height);
    scale = xscale < yscale ? xscale : yscale;

    if (scale)
        jpeg_set_scale_denom(im, scale);
}

int
img_write(FILE *outfile, struct image *im, enum image_type type)
{
    struct writer writer;

    if (type == JPEG)
        jpeg_writer_init(&writer, stream_write_fn, (void*)outfile, im);
    else if (type == PNG)
        png_writer_init(&writer, stream_write_fn, (void*)outfile, im);
    else if (type == PPM)
        if (ppm_writer_init(&writer, stream_write_fn, (void*)outfile, im))
            return -1;

    if (writer.write(&writer))
        return -1;

    writer.free(&writer);
    return 0;
}

int
dumpimage(struct image *im)
{
    long i;
    unsigned char *buf;

    buf = malloc(im->width * sample_size(im->fmt));
    for (i=0; i<im->height; i++)
        if (im->get_scanline(im, buf))
            return -1;
    free(buf);
    return 0;
}

int
format_supports_rgbx(enum image_type out_fmt)
{
    return out_fmt != PPM;
}

int
scale(struct cmd_opt opt)
{
    struct image inimage, yscale, xscale, *last;
    enum image_type in_fmt, out_fmt;
    int random_img, ret, xchange, ychange;
    FILE *file_in, *file_out;
    char *ext;

    random_img = 0;
    file_in = NULL;
    file_out = NULL;

    jpeg_appinit();

    /* Get the output image format by extension so that we can make decisions
     * based on the output image format.
     */
    if (opt.outfile) {
        ext = strrchr(opt.outfile, '.');
        if (!ext || ext_to_image_type(ext, &out_fmt)) {
            printf("Output File extension not recognized.\n");
            return -1;
        }
    }

    if (opt.infile) {
        file_in = fopen(opt.infile, "r");
        if (!file_in) {
            fprintf(stderr, "Unable to open input file.\n");
            return -1;
        }

        if (oil_image_init(&inimage, stream_read_fn, (void *)file_in, &in_fmt))
            return -1;
    } else {
        if (opt.rgbx && (!opt.outfile || format_supports_rgbx(out_fmt)))
            random_init(&inimage, SAMPLE_RGBX, 1000, 1000);
        else
            random_init(&inimage, SAMPLE_RGB, 1000, 1000);
        random_img = 1;
    }

    last = &inimage;

    if (!opt.width && !opt.height) {
        opt.width = inimage.width;
        opt.height = inimage.height;
    }

    if (opt.box)
        fix_ratio(inimage.width, inimage.height, &opt.width, &opt.height);
    else {
        if (!opt.width)
            opt.width = inimage.width;
        if (!opt.height)
            opt.height = inimage.height;
    }

    printf("Resizing from %ldx%ld to %ldx%ld.\n", inimage.width, inimage.height,
        opt.width, opt.height);

    if (!random_img && in_fmt == JPEG) {
        if (opt.prescale) {
            prescale(&inimage, opt.width, opt.height);
            printf("Prescaling to %ldx%ld\n", inimage.width, inimage.height);
        }

        if (opt.rgbx)
            jpeg_set_rgbx(&inimage);
    }

    xchange = inimage.width != opt.width;
    ychange = inimage.height != opt.height;

    if (opt.xy) {
        if (xchange) {
            xscaler_init(&xscale, last, opt.width);
            last = &xscale;
        }
        if (ychange) {
            yscaler_init(&yscale, last, opt.height);
            last = &yscale;
        }
    } else {
        if (ychange) {
            yscaler_init(&yscale, last, opt.height);
            last = &yscale;
        }
        if (xchange) {
            xscaler_init(&xscale, last, opt.width);
            last = &xscale;
        }
    }

    if (opt.outfile) {
        file_out = fopen(opt.outfile, "w");
        if (!file_out) {
            fprintf(stderr, "Unable to open output file.\n");
            return -1;
        }

        ret = img_write(file_out, last, out_fmt);
    } else
        ret = dumpimage(last);

    if (xchange)
        xscale.free(&xscale);
    if (ychange)
        yscale.free(&yscale);

    if (file_out)
        fclose(file_out);
    if (file_in)
        fclose(file_in);

    inimage.free(&inimage);
    return ret;
}

int
main(int argc, char *argv[])
{
    int i;
    char *arg;
    struct cmd_opt opt;

    opt.box = 1;
    opt.rgbx = 0;
    opt.prescale = 1;
    opt.xy = 0;
    opt.width = 0;
    opt.height = 0;
    opt.infile = NULL;
    opt.outfile = NULL;

    for (i=1; i<argc; i++) {
        arg = argv[i];
        if (arg[0] != '-' || arg[1] != '-')
            break;
        if (!strcmp(arg, "--nobox"))
            opt.box = 0;
        else if (!strcmp(arg, "--rgbx"))
            opt.rgbx = 1;
        else if (!strcmp(arg, "--noprescale"))
            opt.prescale = 0;
        else if (!strcmp(arg, "--xy"))
            opt.xy = 1;
        else if (!strcmp(arg, "--width")) {
            if (argc == ++i)
                goto err_exit;
            opt.width = atol(argv[i]);
        }
        else if (!strcmp(arg, "--height")) {
            if (argc == ++i)
                goto err_exit;
            opt.height = atol(argv[i]);
        }
        else if (!strcmp(arg, "--in")) {
            if (argc == ++i)
                goto err_exit;
            opt.infile = argv[i];
        }
        else if (!strcmp(arg, "--out")) {
            if (argc == ++i)
                goto err_exit;
            opt.outfile = argv[i];
        }
        else {
            fprintf(stderr, "Option %s not recognized.\n", arg);
            goto err_exit;
        }
    }

    if (argc != i)
        goto err_exit;

    if (opt.width < 0 || opt.height < 0) {
        fprintf(stderr, "Resize dimensions must be positive numbers or zero.\n");
        return 1;
    }

    if (scale(opt)) {
        fprintf(stderr, "Scaling Failed.\n");
        return 1;
    }

    return 0;

    err_exit:
        fprintf(stderr, "Usage: %s [options]\n", argv[0]);
        return 1;
}
