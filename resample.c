#include "resample.h"
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef __SSE4_1__
#include <emmintrin.h>
#include <smmintrin.h>
#endif

#define TAPS 4

static long
gcd (long a, long b)
{
    long c;
    while (a != 0) {
        c = a;
        a = b%a;
        b = c;
    }
    return b;
}

static int32_t
clamp(int32_t x)
{
    if (x < 0)
        return 0;

    x += 800; /* arbitrary small number to bump up rounding errors */

    if (x & 0x40000000)
        return 0x3FC00000;

    return x;
}

/* Map from a discreet dest coordinate to a continuous source coordinate.
 * The resulting coordinate can range from -0.5 to the maximum of the
 * destination image dimension.
 */
static float
map(long pos, float scale)
{
    return (pos + 0.5) / scale - 0.5;
}

static long
calc_mapping_split(long dim_in, long dim_out, long pos, float *rest)
{
    float scale, smp;
    long smp_i;

    scale = dim_out / (float)dim_in;
    smp = map(pos, scale);
    smp_i = smp < 0 ? -1 : smp;
    *rest = smp - smp_i;
    return smp_i;
}

static long
calc_tap_mult(long dim_in, long dim_out)
{
    if (dim_out > dim_in)
        return 1;
    return dim_in / dim_out;
}

static long
calc_taps(long dim_in, long dim_out)
{
    return calc_tap_mult(dim_in, dim_out) * TAPS;
}

int
sample_size(enum sample_fmt fmt)
{
    switch (fmt) {
    case SAMPLE_GREYSCALE:
        return 1;
        break;
    case SAMPLE_GREYSCALE_ALPHA:
        return 2;
        break;
    case SAMPLE_RGB:
        return 3;
    default:
        return 4;
    }
}

#define rgba_r(x) ((x) & 0x000000FF)
#define rgba_g(x) (((x) & 0x0000FF00) >> 8)
#define rgba_b(x) (((x) & 0x00FF0000) >> 16)
#define rgba_a(x) (((x) & 0xFF000000) >> 24)

static uint32_t
fixed_9_22_to_rgbx(int32_t r, int32_t g, int32_t b)
{
    return (clamp(r) >> 22) +
           ((clamp(g) >> 14) & 0x0000FF00) +
           ((clamp(b) >> 6) & 0x00FF0000);
}

static uint32_t
fixed_9_22_to_rgba(int32_t r, int32_t g, int32_t b, int32_t a)
{
    return fixed_9_22_to_rgbx(r, g, b) + ((clamp(a) << 2) & 0xFF000000);
}

static float
catrom(float x)
{
    if (x<1)
        return (3*x*x*x - 5*x*x + 2) / 2;
    return (-1*x*x*x + 5*x*x - 8*x + 4) / 2;
}

static float
mitchell(float x)
{
    if (x<1)
        return (7*x*x*x - 12*x*x + 5 + 1/3.0) / 6;
    return (-7/3.0*x*x*x + 12*x*x - 20*x + 10 + 2/3.0) / 6;
}

static float
linear(float x)
{
    return 1 - x;
}

static int32_t
f_to_fixed_9_22(float x)
{
    return  x * 4194304;
}

static void
calc_coeffs(int32_t *coeffs, float tx, long tap_mult)
{
    long i, taps;
    float tmp;

    taps = tap_mult * TAPS;
    tx = 1 - tx - taps / 2;

    for (i=0; i<taps; i++) {
        tmp = catrom(fabs(tx) / tap_mult) / tap_mult;
        coeffs[i] = f_to_fixed_9_22(tmp);
        tx += 1;
    }
}

/* Strip */

static long
strip_height(struct strip *st)
{
    return calc_taps(st->in_height, st->out_height);
}

static long
strip_line_len(struct strip *st)
{
    return sample_size(st->fmt) * st->width;
}

void
strip_init(struct strip *st, long in_height, long out_height, long width,
           enum sample_fmt fmt)
{
    long i, line_ary_size, st_height, line_len;

    st->in_height = in_height;
    st->out_height = out_height;

    st_height = strip_height(st);
    line_ary_size = st_height * sizeof(void*);

    st->sl = malloc(line_ary_size);
    st->virt = malloc(line_ary_size);
    st->in_lineno = 0;
    st->out_lineno = 0;
    st->width = width;
    st->fmt = fmt;

    line_len = strip_line_len(st);

    for (i=0; i<st_height; i++) {
        st->sl[i] = malloc(line_len);
        st->virt[i] = st->sl[0];
    }
}

void
strip_free(struct strip *st)
{
    long st_height, i;

    st_height = strip_height(st);

    for (i=0; i<st_height; i++)
        free(st->sl[i]);
    free(st->sl);
    free(st->virt);
}

static void
strip_set_coeffs(struct strip *st, int32_t *coeffs)
{
    float ty;
    long tap_mult;

    calc_mapping_split(st->in_height, st->out_height, st->out_lineno, &ty);
    tap_mult = calc_tap_mult(st->in_height, st->out_height);
    calc_coeffs(coeffs, ty, tap_mult);
}

static void
strip_scale_gen(long len, long height, int32_t *coeffs, unsigned char **sl_in,
                unsigned char *sl_out)
{
    long i, j;
    float result;

    for (i=0; i<len; i++) {
        result = 0;
        for (j=0; j<height; j++)
            result += coeffs[j] * sl_in[j][i];
        sl_out[i] = clamp(result) >> 22;
    }
}

static void
strip_scale_rgba(long width, long height, int32_t *coeffs, uint32_t **sl_in,
                 uint32_t *sl_out)
{
    long i, j;
    float r, g, b, a, coeff;
    uint32_t sample;

    for (i=0; i<width; i++) {
        r = g = b = a = 0;
        for (j=0; j<height; j++) {
            coeff = coeffs[j];
            sample = sl_in[j][i];
            r += coeff * rgba_r(sample);
            g += coeff * rgba_g(sample);
            b += coeff * rgba_b(sample);
            a += coeff * rgba_a(sample);
        }
        sl_out[i] = fixed_9_22_to_rgba(r, g, b, a);
    }
}

static void
strip_scale_rgbx(long width, long height, int32_t *coeffs, uint32_t **sl_in,
                 uint32_t *sl_out)
{
    long i, j;
    int32_t coeff, r, g, b;
    uint32_t sample;

    for (i=0; i<width; i++) {
        r = g = b = 0;
        for (j=0; j<height; j++) {
            coeff = coeffs[j];
            sample = sl_in[j][i];
            r += coeff * rgba_r(sample);
            g += coeff * rgba_g(sample);
            b += coeff * rgba_b(sample);
        }
        sl_out[i] = fixed_9_22_to_rgbx(r, g, b);
    }
}

#ifdef __SSE4_1__
static void
strip_scale_sse(long width, long height, int32_t *coeffs, uint32_t **sl_in,
                     uint32_t *sl_out)
{
    long i, j;
    __m128i sum, pixel, mask, topoff, sample_max;

    topoff = _mm_set1_epi32(800);
    sample_max = _mm_set1_epi32(0x3FC00000);
    mask = _mm_set_epi8(1,1,1,1,1,1,1,1,1,1,1,1,12,8,4,0);

    for (i=0; i<width; i++) {
        sum = _mm_setzero_si128();
        for (j=0; j<height; j++) {
            pixel = _mm_cvtsi32_si128(sl_in[j][i]);
            pixel = _mm_cvtepu8_epi32(pixel);
            pixel = _mm_mullo_epi32(pixel, _mm_set1_epi32(coeffs[j]));
            sum = _mm_add_epi32(pixel, sum);
        }

        sum = _mm_add_epi32(sum, topoff);
        sum = _mm_max_epi32(sum, _mm_setzero_si128());
        sum = _mm_min_epi32(sum, sample_max);
        sum = _mm_srli_epi32(sum, 22);
        sum = _mm_shuffle_epi8(sum, mask);
        sl_out[i] = _mm_cvtsi128_si32(sum);
    }
}
#endif

void
strip_scale(struct strip *st, void *sl_out)
{
    long height;
    int32_t *coeffs;
    void **sl;

    height = strip_height(st);
    coeffs = malloc(height * sizeof(int32_t));
    strip_set_coeffs(st, coeffs);

    sl = st->virt;

    switch (st->fmt) {
#ifdef __SSE4_1__
    case SAMPLE_RGBA:
    case SAMPLE_RGBX:
        strip_scale_sse(st->width, height, coeffs, (uint32_t **)sl,
                        (uint32_t *)sl_out);
        break;
#else
    case SAMPLE_RGBA:
        strip_scale_rgba(st->width, height, coeffs, (uint32_t **)sl,
                         (uint32_t *)sl_out);
        break;
    case SAMPLE_RGBX:
        strip_scale_rgbx(st->width, height, coeffs, (uint32_t **)sl,
                         (uint32_t *)sl_out);
        break;
#endif
    default:
        strip_scale_gen(strip_line_len(st), height, coeffs, (unsigned char **)sl,
                        (unsigned char *)sl_out);
        break;
    }

    st->out_lineno++;
    free(coeffs);
}

/**
 * Advance the strip. Once we hit max_line, we only advance the virtual pointers
 * which basically extends the source image infinitely.
 * Returns a pointer to the bottom scanline in the strip if another line is
 * needed. If we are at the bottom, we return NULL.
 */
static unsigned char*
strip_next(struct strip *st)
{
    long i, cur_pos, st_height;
    void *cur, **virt;

    st_height = strip_height(st);
    virt = st->virt;

    for (i=1; i<st_height; i++)
        virt[i - 1] = virt[i];

    if (st->in_lineno < st->in_height) {
        cur_pos = st->in_lineno%st_height;
        cur = st->sl[cur_pos];
        st->virt[st_height - 1] = cur;
        return cur;
    }

    return NULL;
}

static long
strip_target_pos(struct strip *st)
{
    float smp, _;
    long st_height, smp_i;

    smp = calc_mapping_split(st->in_height, st->out_height, st->out_lineno, &_);
    smp_i = smp < 0 ? -1 : smp;
    st_height = strip_height(st);
    return smp_i + st_height / 2 + 1;
}

void*
strip_next_inbuf(struct strip *st)
{
    long target_pos;
    unsigned char *ybuf;

    target_pos = strip_target_pos(st);

    while (st->in_lineno < target_pos) {
        ybuf = strip_next(st);
        st->in_lineno++;
        if (ybuf)
            return ybuf;
    }

    return NULL;
}

/* Bicubic x scaler */

static void
sample_generic(int cmp, long taps, int32_t *coeffs, unsigned char *in,
               unsigned char *out)
{
    int i;
    long j;
    int32_t result;

    for (i=0; i<cmp; i++) {
        result = 0;
        for (j=0; j<taps; j++)
            result += coeffs[j] * in[j * cmp + i];
        out[i] = clamp(result) >> 22;
    }
}

static uint32_t
sample_rgba(long taps, int32_t *coeffs, uint32_t *in)
{
    long i;
    float r, g, b, a, coeff;
    uint32_t sample;

    r = g = b = a = 0;
    for (i=0; i<taps; i++) {
        coeff = coeffs[i];
        sample = in[i];
        r += coeff * rgba_r(sample);
        g += coeff * rgba_g(sample);
        b += coeff * rgba_b(sample);
        a += coeff * rgba_a(sample);
    }
    return fixed_9_22_to_rgba(r, g, b, a);
}

static uint32_t
sample_rgbx(long taps, int32_t *coeffs, uint32_t *in)
{
    long i;
    int32_t r, g, b, coeff;
    uint32_t sample;

    r = g = b = 0;
    for (i=0; i<taps; i++) {
        coeff = coeffs[i];
        sample = in[i];
        r += coeff * rgba_r(sample);
        g += coeff * rgba_g(sample);
        b += coeff * rgba_b(sample);
    }
    return fixed_9_22_to_rgbx(r, g, b);
}

#ifdef __SSE4_1__
static uint32_t
sample_sse(long taps, int32_t *coeffs, uint32_t *in)
{
    long i;
    __m128i sum, pixel, mask;

    sum = _mm_setzero_si128();
    for (i=0; i<taps; i++) {
        pixel = _mm_cvtsi32_si128(in[i]);
        pixel = _mm_cvtepu8_epi32(pixel);
        pixel = _mm_mullo_epi32(pixel, _mm_set1_epi32(coeffs[i]));
        sum = _mm_add_epi32(pixel, sum);
    }

    sum = _mm_add_epi32(sum, _mm_set1_epi32(800));
    sum = _mm_max_epi32(sum, _mm_setzero_si128());
    sum = _mm_min_epi32(sum, _mm_set1_epi32(0x3FC00000));
    sum = _mm_srli_epi32(sum, 22);
    mask = _mm_set_epi8(1,1,1,1,1,1,1,1,1,1,1,1,12,8,4,0);
    sum = _mm_shuffle_epi8(sum, mask);
    return _mm_cvtsi128_si32(sum);
}
#endif

static void
xscale_set_sample(enum sample_fmt fmt, long taps, int32_t *coeffs, void *in,
                  void *out)
{
    switch (fmt) {
#ifdef __SSE4_1__
    case SAMPLE_RGBA:
    case SAMPLE_RGBX:
        *(uint32_t *)out = sample_sse(taps, coeffs, (uint32_t *)in);
        break;
#else
    case SAMPLE_RGBA:
        *(uint32_t *)out = sample_rgba(taps, coeffs, (uint32_t *)in);
        break;
    case SAMPLE_RGBX:
        *(uint32_t *)out = sample_rgbx(taps, coeffs, (uint32_t *)in);
        break;
#endif
    default:
        sample_generic(sample_size(fmt), taps, coeffs, in, out);
        break;
    }
}

static void
xscale_gcd(void *in, long in_width, void *out, long out_width,
           enum sample_fmt fmt) {
    float tx;
    int32_t *coeffs;
    long i, j, xsmp_i, in_chunk, out_chunk, scale_gcd, taps, tap_mult;
    char *in_pos, *out_pos;
    int cmp;

    tap_mult = calc_tap_mult(in_width, out_width);
    taps = tap_mult * TAPS;
    coeffs = malloc(taps * sizeof(int32_t));

    scale_gcd = gcd(in_width, out_width);
    in_chunk = in_width / scale_gcd;
    out_chunk = out_width / scale_gcd;
    cmp = sample_size(fmt);

    for (i=0; i<out_chunk; i++) {
        xsmp_i = calc_mapping_split(in_width, out_width, i, &tx);
        calc_coeffs(coeffs, tx, tap_mult);

        in_pos = (char *)in + (xsmp_i + 1 - taps / 2) * cmp;
        out_pos = (char *)out + i * cmp;

        for (j=0; j<scale_gcd; j++) {
            xscale_set_sample(fmt, taps, coeffs, in_pos, out_pos);
            in_pos += in_chunk * cmp;
            out_pos += out_chunk * cmp;
        }
    }

    free(coeffs);
}

/* padded scanline */

static long
padded_sl_padwidth(struct padded_sl *psl)
{
    return calc_taps(psl->in_width, psl->out_width) + 1;
}

void
padded_sl_init(struct padded_sl *psl, long in_width, long out_width,
               enum sample_fmt fmt)
{
    long pad_width, buf_size;
    int cmp;

    psl->in_width = in_width;
    psl->out_width = out_width;
    psl->fmt = fmt;

    cmp = sample_size(fmt);
    pad_width = padded_sl_padwidth(psl);
    buf_size = (in_width + 2 * pad_width) * cmp;

    psl->pad_left = malloc(buf_size);
    psl->buf = (unsigned char *)psl->pad_left + pad_width * cmp;
}

void
padded_sl_free(struct padded_sl *psl)
{
    free(psl->pad_left);
}

/**
 * pad points to the first byte in the pad area.
 * src points to the sample that will be replicated in the pad area.
 * width is the number of samples in the pad area.
 * cmp is the number of components per sample.
 */
static void
padded_sl_pad(unsigned char *pad, unsigned char *src, int width, int cmp)
{
    int i, j;

    for (i=0; i<width; i++)
        for (j=0; j<cmp; j++)
            pad[i * cmp + j] = src[j];
}

static void
padded_sl_extend_edges(struct padded_sl *psl)
{
    int cmp;
    long pad_width;
    unsigned char *pad_right;

    cmp = sample_size(psl->fmt);
    pad_width = padded_sl_padwidth(psl);
    pad_right = (unsigned char *)psl->buf + psl->in_width * cmp;

    padded_sl_pad(psl->pad_left, psl->buf, pad_width, cmp);
    padded_sl_pad(pad_right, pad_right - cmp, pad_width, cmp);
}

void
padded_sl_scale(struct padded_sl *psl, unsigned char *out)
{
    padded_sl_extend_edges(psl);
    xscale_gcd(psl->buf, psl->in_width, out, psl->out_width, psl->fmt);
}
