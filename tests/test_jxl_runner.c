/*
 * Direct unit test for the cancellable parallel runner (oil_jxl_threads), with
 * no libjxl decode in the loop. Drives oil_libjxl_parallel_runner with a
 * synthetic init/func and asserts:
 *   - every index in [start,end) runs exactly once across the worker pool;
 *   - dispatch works immediately after create (the gen-0 startup path, where a
 *     worker that hasn't run yet must still pick up the first job);
 *   - cancel makes a dispatch return an error, and reset restores it.
 *
 * The full multithreaded decode path is covered by test_jxl_cancel /
 * test_jxl_regress; this isolates the runner's dispatch/cancel/reset logic.
 */

#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <jxl/parallel_runner.h>
#include "oil_jxl_threads.h"

#define N        1000
#define NTHREADS 4

static _Atomic int counts[N];
static _Atomic int max_tid;

static int test_init(void *jxl, size_t num_threads)
{
	(void)jxl;
	assert(num_threads == NTHREADS);
	return 0;
}

static void test_func(void *jxl, uint32_t i, size_t tid)
{
	int prev;
	(void)jxl;
	assert(i < N);
	assert(tid < NTHREADS);
	atomic_fetch_add_explicit(&counts[i], 1, memory_order_relaxed);
	prev = atomic_load_explicit(&max_tid, memory_order_relaxed);
	while ((int)tid > prev && !atomic_compare_exchange_weak_explicit(
	        &max_tid, &prev, (int)tid,
	        memory_order_relaxed, memory_order_relaxed))
		;
}

static void reset_counts(void)
{
	int i;
	for (i = 0; i < N; i++)
		atomic_store_explicit(&counts[i], 0, memory_order_relaxed);
}

static void assert_each_once(void)
{
	int i;
	for (i = 0; i < N; i++)
		assert(atomic_load_explicit(&counts[i], memory_order_relaxed) == 1);
}

int main(void)
{
	void *r;
	JxlParallelRetCode rc;

	printf("oil_jxl_threads runner:\n");

	r = oil_libjxl_runner_create(NTHREADS);
	assert(r);

	/* Dispatch immediately after create: gen-0 startup path. */
	reset_counts();
	rc = oil_libjxl_parallel_runner(r, NULL, test_init, test_func, 0, N);
	assert(rc == 0);
	assert_each_once();
	printf("  dispatch after create: every index ran once\n");

	/* Subsequent generation still covers the full range. */
	reset_counts();
	rc = oil_libjxl_parallel_runner(r, NULL, test_init, test_func, 0, N);
	assert(rc == 0);
	assert_each_once();
	printf("  re-dispatch: every index ran once\n");

	/* Cancelled: the next dispatch returns an error. */
	oil_libjxl_runner_cancel(r);
	rc = oil_libjxl_parallel_runner(r, NULL, test_init, test_func, 0, N);
	assert(rc != 0);
	printf("  cancel: dispatch returns error\n");

	/* Reset restores normal dispatch. */
	oil_libjxl_runner_reset(r);
	reset_counts();
	rc = oil_libjxl_parallel_runner(r, NULL, test_init, test_func, 0, N);
	assert(rc == 0);
	assert_each_once();
	printf("  reset: dispatch works again\n");

	/* Informational: how many distinct workers were observed. Not asserted --
	 * work distribution across the pool is a scheduling property, not a
	 * correctness one (a single worker draining the cursor is still correct). */
	printf("  max worker id observed: %d (of %d)\n",
		atomic_load_explicit(&max_tid, memory_order_relaxed), NTHREADS);

	oil_libjxl_runner_destroy(r);
	printf("all runner tests pass\n");
	return 0;
}
