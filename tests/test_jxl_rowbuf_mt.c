/*
 * Multithreaded stress test for oil_jxl_rowbuf driven through the real pthreads
 * condvar waiter (oil_jxl_threads). Several producer threads feed a partition
 * of the rows concurrently while the main thread consumes top-to-bottom; small
 * back-pressure windows force producers to park, and reverse-order configs
 * force the consumer-starvation backstop (cap lifting) to fire.
 *
 * Each config is run many times to shake out lost-wakeup / ordering races. The
 * assertions are: every row comes out byte-exact (px oracle), and the run
 * completes at all (a deadlock would hang -- run under a timeout, and under
 * ThreadSanitizer to catch data races the assertions can't see).
 */

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oil_jxl_rowbuf.h"
#include "oil_jxl_threads.h"

static unsigned char px(size_t gx, size_t gy, size_t c)
{
	return (unsigned char)((gx * 131u + gy * 1313u + c * 7u + 17u) & 0xffu);
}

/* Feed one full row, split into three segments delivered right-to-left, so
 * tiles within a row also arrive out of order. */
static void feed_row(struct oil_jxl_rowbuf *rb, size_t y, size_t w, size_t bpp)
{
	size_t third = w / 3 ? w / 3 : 1;
	size_t bounds[4];
	int k;
	bounds[0] = 0;
	bounds[1] = third;
	bounds[2] = 2 * third;
	bounds[3] = w;
	for (k = 3; k >= 1; k--) {
		size_t x = bounds[k - 1];
		size_t n = bounds[k] - bounds[k - 1];
		unsigned char *buf;
		size_t i, c;
		if (n == 0)
			continue;
		buf = malloc(n * bpp);
		assert(buf);
		for (i = 0; i < n; i++)
			for (c = 0; c < bpp; c++)
				buf[i * bpp + c] = px(x + i, y, c);
		oil_jxl_rowbuf_write_segment(rb, x, y, n, buf);
		free(buf);
	}
}

struct prod_arg {
	struct oil_jxl_rowbuf *rb;
	size_t w, h, bpp, stride, id;
	int reverse;
};

static void *producer(void *a)
{
	struct prod_arg *p = a;
	size_t cap = p->h / p->stride + 1, cnt = 0, y, i;
	size_t *rows = malloc(cap * sizeof(*rows));
	assert(rows);
	for (y = p->id; y < p->h; y += p->stride)
		rows[cnt++] = y;
	if (p->reverse)
		for (i = cnt; i-- > 0; )
			feed_row(p->rb, rows[i], p->w, p->bpp);
	else
		for (i = 0; i < cnt; i++)
			feed_row(p->rb, rows[i], p->w, p->bpp);
	free(rows);
	return NULL;
}

static void run_config(size_t P, size_t w, size_t h, size_t bpp, int reverse,
	const char *window)
{
	struct oil_jxl_waiter *waiter;
	struct oil_jxl_rowbuf *rb;
	pthread_t *th;
	struct prod_arg *args;
	size_t i, y, j, c;

	setenv("OIL_JXL_WINDOW", window, 1);   /* before create */
	waiter = oil_jxl_condvar_waiter_create();
	assert(waiter);
	rb = oil_jxl_rowbuf_create(0, 0, w, h, bpp, 32, waiter);
	assert(rb);

	th = malloc(P * sizeof(*th));
	args = malloc(P * sizeof(*args));
	assert(th && args);
	for (i = 0; i < P; i++) {
		args[i].rb = rb;
		args[i].w = w;
		args[i].h = h;
		args[i].bpp = bpp;
		args[i].stride = P;
		args[i].id = i;
		args[i].reverse = reverse;
		assert(pthread_create(&th[i], NULL, producer, &args[i]) == 0);
	}

	/* Consume top-to-bottom on this thread. */
	for (y = 0; y < h; y++) {
		unsigned char *row = oil_jxl_rowbuf_wait_row(rb, y);
		assert(row);
		for (j = 0; j < w; j++)
			for (c = 0; c < bpp; c++)
				assert(row[j * bpp + c] == px(j, y, c));
		oil_jxl_rowbuf_release_row(rb, y);
	}

	for (i = 0; i < P; i++)
		pthread_join(th[i], NULL);

	oil_jxl_rowbuf_destroy(rb);
	oil_jxl_condvar_waiter_destroy(waiter);
	free(th);
	free(args);
}

int main(int argc, char **argv)
{
	/* Fewer iterations by default; pass a count (e.g. small under TSan). */
	int iters = argc > 1 ? atoi(argv[1]) : 30;
	int it;

	printf("oil_jxl_rowbuf multithreaded (%d iters/config):\n", iters);
	for (it = 0; it < iters; it++) {
		run_config(4, 200, 300, 3, 0, "4");   /* heavy back-pressure */
		run_config(8,  64, 500, 4, 0, "8");   /* many threads, narrow */
		run_config(1, 100,  64, 3, 1, "4");   /* reverse: backstop */
		run_config(4, 120, 200, 2, 1, "4");   /* reverse + concurrency */
		run_config(4, 200, 300, 3, 0, "0");   /* unbounded: pure concurrency */
	}
	printf("all multithreaded rowbuf tests passed\n");
	return 0;
}
