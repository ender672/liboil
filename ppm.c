#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

#include "oil.h"

struct ppm {
    read_fn_t read;
    void *read_ctx;
    unsigned char *buf;
    size_t buflen;
    unsigned char *cur;
};

/* The header must fit in PPM_BUFSIZE. A valid ppm could have enough whitespace
 * to exceed PPM_BUFSIZE, but doesn't seem worth the effort to support this.
*/
#define PPM_BUFSIZE 1024
#define PPM_MAX_DIM 100000

static void
ppm_advance(struct ppm *ppm, size_t n)
{
    ppm->cur += n;
    ppm->buflen -= n;
}

static int
ppm_read(struct ppm *ppm, size_t n, unsigned char *sl, size_t *read_len)
{
    return ppm->read(ppm->read_ctx, n, sl, read_len);
}

static int
ppm_get_scanline(struct image *image, unsigned char *sl)
{
    struct ppm *ppm;
    size_t n, read_len;

    ppm = (struct ppm *)image->data;
    n = image->width * sample_size(image->fmt);

    if (ppm->buflen >= n) {
	memcpy(sl, ppm->cur, n);
	ppm_advance(ppm, n);
	return 0;
    } else if (ppm->buflen) {
	memcpy(sl, ppm->cur, ppm->buflen);
	n -= ppm->buflen;
	sl += ppm->buflen;
	ppm->buflen = 0;
    }

    if (ppm_read(ppm, n, sl, &read_len))
	return -1;

    return read_len == n ? 0 : -1;
}

static int
ppm_parse_header(struct ppm *ppm, long *w, long *h, long *max, int sig_bytes)
{
    int len, match_count, ver;
    char *cur;

    cur = (char *)ppm->cur;

    switch (sig_bytes) {
      case 0:
	match_count = sscanf(cur, "P%d%ld%ld%ld%n", &ver, w, h, max, &len);
	if (match_count < 4 || ver != 6 || !isspace(cur[len]))
	    return -1;
	break;
      case 1:
	match_count = sscanf(cur, "%d%ld%ld%ld%n", &ver, w, h, max, &len);
	if (match_count < 4 || ver != 6 || !isspace(cur[len]))
	    return -1;
	break;
      case 2:
	if (!isspace(cur[0]))
	    return -1;
	match_count = sscanf(cur, "%ld%ld%ld%n", w, h, max, &len);
	if (match_count < 3 || !isspace(cur[len]))
	    return -1;
	break;
      default:
	return -1;
    }

    ppm_advance(ppm, len + 1);
    return 0;
}

static void
ppm_free(struct image *image)
{
    struct ppm *ppm;

    ppm = (struct ppm *)image->data;
    free(ppm->buf);
    free(ppm);
}

/* error codes:
 * -1: io read failed
 * -2: bad header
 * -3: unsupported width
 * -4: unsupported height
 * -5: unsupported max color value (currently only 255 is supported)
 * -6: sig_bytes 
 */
int
ppm_init(struct image *image, read_fn_t read, void *ctx, int sig_bytes)
{
    struct ppm *ppm;
    long max;

    image->free = ppm_free;
    ppm = image->data = malloc(sizeof(struct ppm));
    ppm->buf = ppm->cur = malloc(PPM_BUFSIZE + 1);

    ppm->read = read;
    ppm->read_ctx = ctx;
    if (ppm_read(ppm, PPM_BUFSIZE, ppm->buf, &ppm->buflen))
	return -1;

    ppm->cur[ppm->buflen] = 0;
    image->fmt = SAMPLE_RGB;
    image->get_scanline = ppm_get_scanline;

    if (ppm_parse_header(ppm, &image->width, &image->height, &max, sig_bytes))
	return -2;

    if (image->width < 1 || image->width > PPM_MAX_DIM)
	return -3;

    if (image->height < 1 || image->height > PPM_MAX_DIM)
	return -4;

    if (max != 255)
	return -5;

    return 0;
}

/* PPM Writer */

void
ppm_writer_free(struct writer *writer)
{
    free(writer->data);
}

int
ppm_writer_write(struct writer *writer)
{
    int header_len;
    long i, sl_len;
    struct image *src;
    unsigned char *buf;

    buf = (unsigned char *)writer->data;
    src = writer->src;

    header_len = snprintf((char *)buf, PPM_BUFSIZE, "P6 %ld %ld 255 ",
                          src->width, src->height);
    if (header_len < 0)
        return -1;

    if (writer->write_cb(writer->ctx, buf, (size_t)header_len))
	return -1;

    sl_len = src->width * 3;
    for (i=0; i<src->height; i++) {
        if (src->get_scanline(src, buf))
            return -1;

        if (writer->write_cb(writer->ctx, buf, sl_len))
            return -1;
    }
    return 0;
}

int
ppm_writer_init(struct writer *writer, write_fn_t write_fn, void *ctx,
		struct image *src)
{
    long sl_width;

    if (sample_size(src->fmt) != 3)
        return -1;

    sl_width = src->width * 3;
    writer->src = src;
    writer->data = malloc(PPM_BUFSIZE > sl_width ? PPM_BUFSIZE : sl_width);
    writer->write = ppm_writer_write;
    writer->free = ppm_writer_free;
    writer->write_cb = write_fn;
    writer->ctx = ctx;

    return 0;
}
