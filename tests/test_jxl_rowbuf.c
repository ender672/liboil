/*
 * Direct unit tests for the out-of-order scanline reorder buffer
 * (oil_jxl_rowbuf), with no libjxl in the loop. Feeds scripted and
 * seeded-random partial-segment delivery orders that a real decode could never
 * be coerced into producing, and asserts the finalized rows are byte-exact.
 *
 * This is the behavioral oracle that must survive the atomics+waiter rewrite of
 * the reorder buffer unchanged.
 *
 * Single-threaded by construction: OIL_JXL_WINDOW is pinned (see main) so the
 * back-pressure window equals the height and a producer never parks waiting for
 * a consumer that isn't running. A trivial no-op waiter is supplied -- valid
 * because only one thread ever touches the buffer -- and its wait() asserts, so
 * any unexpected block fails loudly instead of hanging. Concurrency and
 * back-pressure under a real (condvar) waiter are exercised by
 * test_jxl_rowbuf_mt.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oil_jxl_rowbuf.h"
#include "oil_jxl_waiter.h"

/* No-op waiter: legal only single-threaded. lock/unlock/wake do nothing;
 * wait must never be reached (rows are always ready when waited). */
static void noop_lock(void *o)   { (void)o; }
static void noop_unlock(void *o) { (void)o; }
static void noop_wait(void *o, int ch)
{
	(void)o; (void)ch;
	assert(0 && "single-threaded rowbuf test must never block");
}
static void noop_wake(void *o, int ch, int all) { (void)o; (void)ch; (void)all; }
static const struct oil_jxl_waiter noop_waiter = {
	noop_lock, noop_unlock, noop_wait, noop_wake, NULL
};

static struct oil_jxl_rowbuf *rb_new(size_t x0, size_t y0, size_t w, size_t h,
	size_t bpp, size_t tile_w)
{
	return oil_jxl_rowbuf_create(x0, y0, w, h, bpp, tile_w, &noop_waiter);
}

/* Deterministic per-pixel content keyed on full-image (gx,gy) and channel. */
static unsigned char px(size_t gx, size_t gy, size_t c)
{
	return (unsigned char)((gx * 131u + gy * 1313u + c * 7u + 17u) & 0xffu);
}

/* Feed one segment of @n pixels at full-image (@x,@y), filled per px(). */
static void feed(struct oil_jxl_rowbuf *rb, size_t x, size_t y, size_t n,
	size_t bpp)
{
	unsigned char *buf = malloc(n * bpp);
	size_t i, c;
	assert(buf);
	for (i = 0; i < n; i++)
		for (c = 0; c < bpp; c++)
			buf[i * bpp + c] = px(x + i, y, c);
	oil_jxl_rowbuf_write_segment(rb, x, y, n, buf);
	free(buf);
}

/* Assert a finalized crop-local row (full-image row @gy) matches px() over the
 * crop columns [x0, x0+w). */
static void check_row(const unsigned char *row, size_t x0, size_t gy,
	size_t w, size_t bpp)
{
	size_t j, c;
	for (j = 0; j < w; j++)
		for (c = 0; c < bpp; c++)
			assert(row[j * bpp + c] == px(x0 + j, gy, c));
}

/* Pull every row top-to-bottom from a full-image-cropped rowbuf and verify. */
static void drain_and_check(struct oil_jxl_rowbuf *rb, size_t x0, size_t y0,
	size_t w, size_t h, size_t bpp)
{
	size_t y;
	for (y = 0; y < h; y++) {
		unsigned char *row = oil_jxl_rowbuf_wait_row(rb, y);
		assert(row);
		check_row(row, x0, y0 + y, w, bpp);
		oil_jxl_rowbuf_release_row(rb, y);
	}
}

static void test_full_rows_inorder(void)
{
	size_t w = 600, h = 40, bpp = 3, y;
	struct oil_jxl_rowbuf *rb = rb_new(0, 0, w, h, bpp, 256);
	assert(rb);
	for (y = 0; y < h; y++)
		feed(rb, 0, y, w, bpp);
	drain_and_check(rb, 0, 0, w, h, bpp);
	oil_jxl_rowbuf_destroy(rb);
	printf("  full rows, in order: ok\n");
}

static void test_reverse_rows(void)
{
	size_t w = 137, h = 50, bpp = 4, y;
	struct oil_jxl_rowbuf *rb = rb_new(0, 0, w, h, bpp, 16);
	assert(rb);
	for (y = h; y-- > 0; )           /* deliver rows bottom-to-top */
		feed(rb, 0, y, w, bpp);
	drain_and_check(rb, 0, 0, w, h, bpp);
	oil_jxl_rowbuf_destroy(rb);
	printf("  rows in reverse: ok\n");
}

/* Each row split into tiles delivered in a scrambled order, with some segments
 * straddling tile boundaries. */
static void test_out_of_order_tiles(void)
{
	size_t w = 100, h = 8, bpp = 2, tile_w = 8, y;
	struct oil_jxl_rowbuf *rb = rb_new(0, 0, w, h, bpp, tile_w);
	assert(rb);
	for (y = 0; y < h; y++) {
		/* A partition of [0,100) into tile-straddling chunks, delivered in a
		 * scrambled order. Each column is covered exactly once, so every tile
		 * fills exactly and the row completes regardless of arrival order. */
		size_t starts[] = { 45, 0, 78, 21, 96, 60,  7, 90, 33 };
		size_t ns[]     = { 15, 7, 12, 12,  4, 18, 14,  6, 12 };
		size_t i, covered = 0;
		for (i = 0; i < sizeof(ns) / sizeof(ns[0]); i++)
			covered += ns[i];
		assert(covered == w);
		for (i = 0; i < sizeof(starts) / sizeof(starts[0]); i++)
			feed(rb, starts[i], y, ns[i], bpp);
	}
	drain_and_check(rb, 0, 0, w, h, bpp);
	oil_jxl_rowbuf_destroy(rb);
	printf("  out-of-order straddling tiles: ok\n");
}

/* Crop interior of a larger image: segments overhang the crop on every side and
 * land on out-of-crop rows; only the in-crop rect must be buffered. */
static void test_crop_clipping(void)
{
	size_t x0 = 5, y0 = 3, w = 20, h = 10, bpp = 3, tile_w = 4;
	struct oil_jxl_rowbuf *rb = rb_new(x0, y0, w, h, bpp, tile_w);
	size_t gy;
	assert(rb);
	/* Out-of-crop rows above and below: must be ignored, not buffered. */
	feed(rb, 0, y0 - 1, x0 + w + 5, bpp);
	feed(rb, 0, y0 + h, x0 + w + 5, bpp);
	for (gy = y0; gy < y0 + h; gy++)
		/* One wide segment overhanging the crop by 3px left and 3px right. */
		feed(rb, x0 - 3, gy, w + 6, bpp);
	drain_and_check(rb, x0, y0, w, h, bpp);
	oil_jxl_rowbuf_destroy(rb);
	printf("  crop clipping (overhang + out-of-range rows): ok\n");
}

static void test_try_row(void)
{
	size_t w = 4, bpp = 1;
	struct oil_jxl_rowbuf *rb = rb_new(0, 0, w, 2, bpp, 4);
	unsigned char *row;
	assert(rb);
	assert(oil_jxl_rowbuf_try_row(rb, 0) == NULL);   /* nothing written yet */
	feed(rb, 0, 0, w, bpp);
	row = oil_jxl_rowbuf_try_row(rb, 0);
	assert(row);
	check_row(row, 0, 0, w, bpp);
	assert(oil_jxl_rowbuf_try_row(rb, 1) == NULL);   /* row 1 still pending */
	oil_jxl_rowbuf_release_row(rb, 0);
	feed(rb, 0, 1, w, bpp);
	row = oil_jxl_rowbuf_try_row(rb, 1);
	assert(row);
	check_row(row, 0, 1, w, bpp);
	oil_jxl_rowbuf_release_row(rb, 1);
	oil_jxl_rowbuf_destroy(rb);
	printf("  try_row readiness: ok\n");
}

static void test_abort(void)
{
	size_t w = 8, bpp = 3;
	struct oil_jxl_rowbuf *rb = rb_new(0, 0, w, 4, bpp, 4);
	assert(rb);
	feed(rb, 0, 0, 4, bpp);          /* fills tile 0 only; row 0 incomplete */
	assert(oil_jxl_rowbuf_try_row(rb, 0) == NULL);
	oil_jxl_rowbuf_abort(rb);
	assert(oil_jxl_rowbuf_wait_row(rb, 0) == NULL);  /* aborted, not finalized */
	assert(oil_jxl_rowbuf_wait_row(rb, 1) == NULL);
	oil_jxl_rowbuf_destroy(rb);     /* must still free the dangling tile */
	printf("  abort releases waiters: ok\n");
}

struct seg { size_t x, y, n; };

/* For each seed: random dimensions, each row split into random-width segments,
 * all segments shuffled across the whole image, fed, then verified. Catches
 * ordering bugs the hand-written cases miss. */
static void test_random_oracle(void)
{
	unsigned seed;
	for (seed = 1; seed <= 40; seed++) {
		struct oil_jxl_rowbuf *rb;
		struct seg *segs = NULL;
		size_t cap = 0, cnt = 0, w, h, bpp, tile_w, y, i;
		srand(seed);
		w      = 1 + (size_t)(rand() % 200);
		h      = 1 + (size_t)(rand() % 120);
		bpp    = 1 + (size_t)(rand() % 4);
		tile_w = 1 + (size_t)(rand() % 64);

		for (y = 0; y < h; y++) {
			size_t col = 0;
			while (col < w) {
				size_t rem = w - col;
				size_t n = 1 + (size_t)(rand() % 17);
				if (n > rem) n = rem;
				if (cnt == cap) {
					cap = cap ? cap * 2 : 256;
					segs = realloc(segs, cap * sizeof(*segs));
					assert(segs);
				}
				segs[cnt].x = col;
				segs[cnt].y = y;
				segs[cnt].n = n;
				cnt++;
				col += n;
			}
		}
		/* Fisher-Yates shuffle of the global delivery order. */
		for (i = cnt; i-- > 0; ) {
			size_t j = (size_t)(rand() % (int)(i + 1));
			struct seg t = segs[i];
			segs[i] = segs[j];
			segs[j] = t;
		}

		rb = rb_new(0, 0, w, h, bpp, tile_w);
		assert(rb);
		for (i = 0; i < cnt; i++)
			feed(rb, segs[i].x, segs[i].y, segs[i].n, bpp);
		drain_and_check(rb, 0, 0, w, h, bpp);
		oil_jxl_rowbuf_destroy(rb);
		free(segs);
	}
	printf("  seeded-random orderings (40 seeds): ok\n");
}

int main(void)
{
	/* Pin the back-pressure window to the full height so single-threaded
	 * write-all-then-read never parks a producer (would assert in noop_wait). */
	setenv("OIL_JXL_WINDOW", "0", 1);

	printf("oil_jxl_rowbuf:\n");
	test_full_rows_inorder();
	test_reverse_rows();
	test_out_of_order_tiles();
	test_crop_clipping();
	test_try_row();
	test_abort();
	test_random_oracle();
	printf("all rowbuf tests passed\n");
	return 0;
}
