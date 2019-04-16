#include "oil_resample.h"
#include "oil_libjpeg.h"
#include "oil_libpng.h"
#include <gtk/gtk.h>
#include <jpeglib.h>
#include <png.h>
#include <pthread.h>
#include <semaphore.h>

struct img_decode_context {
	GtkWidget *darea;
	cairo_surface_t *sfc_complete;
	cairo_surface_t *sfc;
	char *path;
	int width;
	int height;
	pthread_t worker_thread;
	sem_t surface_set_sem;
	guint redraw_tag;
	guint header_read_tag;
	guint decode_complete_tag;
};

void spawn_decoder(struct img_decode_context *ctx);
GtkWidget *create_window(GtkApplication *app, struct img_decode_context *ctx);

static void translate(unsigned char *in, unsigned char *out, int width, enum oil_colorspace cs)
{
	int i;

	switch(cs) {
	case OIL_CS_G:
		for (i=0; i<width; i++) {
			out[0] = out[1] = out[2] = in[0];
			out[3] = 0xFF;
			in += 1;
			out += 4;
		}
		break;
	case OIL_CS_GA:
		for (i=0; i<width; i++) {
			out[0] = out[1] = out[2] = in[0];
			out[3] = in[1];
			in += 2;
			out += 4;
		}
		break;
	case OIL_CS_RGB:
		for (i=0; i<width; i++) {
			out[0] = in[2];
			out[1] = in[1];
			out[2] = in[0];
			out[3] = 0xFF;
			in += 3;
			out += 4;
		}
		break;
	case OIL_CS_RGBA:
		for (i=0; i<width; i++) {
			out[0] = in[2];
			out[1] = in[1];
			out[2] = in[0];
			out[3] = in[3];
			in += 4;
			out += 4;
		}
		break;
	default:
		break;
	}
}

static void fix_ratio_shrink_only(int in_width, int in_height, int *out_width,
	int *out_height)
{
	oil_fix_ratio(in_width, in_height, out_width, out_height);
	if (*out_width > in_width || *out_height > in_height) {
		*out_width = in_width;
		*out_height = in_height;
	}
}

static void cleanup_auto_redraw(struct img_decode_context *ctx)
{
	if (ctx->redraw_tag) {
		g_source_remove(ctx->redraw_tag);
		ctx->redraw_tag = 0;
	}
}

static gboolean on_draw_event(GtkWidget *widget, cairo_t *cr, gpointer user_data)
{	struct img_decode_context *ctx;
	cairo_surface_t *sfc;
	ctx = (struct img_decode_context *)user_data;
	sfc = ctx->sfc_complete ? ctx->sfc_complete : ctx->sfc;
	cairo_set_source_surface(cr, sfc, 0, 0);
	cairo_paint(cr);
	return TRUE;
}

static int redraw(void *arg)
{
	struct img_decode_context *ctx;

	ctx = (struct img_decode_context *)arg;
	if (!ctx->worker_thread) {
		ctx->redraw_tag = 0;
		return FALSE;
	}
	cairo_surface_mark_dirty(ctx->sfc);
	gtk_widget_queue_draw(ctx->darea);
	return TRUE;
}

void kill_decoder(struct img_decode_context *ctx)
{
	if (ctx->worker_thread) {
		pthread_cancel(ctx->worker_thread);
		pthread_join(ctx->worker_thread, NULL);
		ctx->worker_thread = 0;
	}

	if (ctx->decode_complete_tag) {
		g_source_remove(ctx->decode_complete_tag);
		ctx->decode_complete_tag = 0;
	}
}

static gboolean on_destroy_handler(GtkWidget *widget, gpointer user_data)
{
	struct img_decode_context *ctx;
	ctx = (struct img_decode_context *)user_data;

	cleanup_auto_redraw(ctx);
	kill_decoder(ctx);
	if (ctx->sfc) {
		cairo_surface_destroy(ctx->sfc);
		ctx->sfc = NULL;
	}
	if (ctx->sfc_complete) {
		cairo_surface_destroy(ctx->sfc_complete);
		ctx->sfc_complete = NULL;
	}
	if (ctx->path) {
		g_free(ctx->path);
		ctx->path = NULL;
	}

	sem_destroy(&ctx->surface_set_sem);
	free(ctx);

	return TRUE;
}

static cairo_surface_t *create_surface(GtkWidget *widget, int width, int height)
{
	cairo_surface_t *sfc;
	int hidpi_scale, w_width, w_height;
	hidpi_scale = gtk_widget_get_scale_factor(widget);
	w_width = width * hidpi_scale;
	w_height = height * hidpi_scale;
	sfc = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, w_width, w_height);
	cairo_surface_set_device_scale(sfc, hidpi_scale, hidpi_scale);
	return sfc;
}

static gboolean on_configure_event(GtkWidget *widget, GdkEventConfigure *event, gpointer user_data)
{
	struct img_decode_context *ctx;
	cairo_surface_t *sfc;

	ctx = (struct img_decode_context *)user_data;
	sfc = create_surface(widget, event->width, event->height);

	if (ctx->sfc) {
		kill_decoder(ctx);
		cairo_surface_destroy(ctx->sfc);
		ctx->sfc = sfc;
		spawn_decoder(ctx);
	} else if (ctx->sfc_complete) {
		ctx->sfc = sfc;
		spawn_decoder(ctx);
	} else {
		ctx->sfc = sfc;
		sem_post(&ctx->surface_set_sem);
		ctx->redraw_tag = g_timeout_add_seconds(1, redraw, ctx);
	}

	return TRUE;
}

void get_monitor_geom(GtkWidget *window, int *width, int *height)
{
	GdkMonitor *monitor;
	GdkRectangle rect;
	GdkDisplay *display;

	display = gtk_widget_get_display(window);
	monitor = gdk_display_get_monitor(display, 0);
	gdk_monitor_get_geometry(monitor, &rect);
	*width = rect.width;
	*height = rect.height;
}

static int header_read(void *arg)
{
	GtkWidget *window;
	int width, height, hidpi_scale;
	struct img_decode_context *ctx;

	ctx = (struct img_decode_context *)arg;

	hidpi_scale = gtk_widget_get_scale_factor(ctx->darea);
	window = gtk_widget_get_toplevel(ctx->darea);
	get_monitor_geom(window, &width, &height);
	width *= 0.7 * hidpi_scale;
	height *= 0.8 * hidpi_scale;
	fix_ratio_shrink_only(ctx->width, ctx->height, &width, &height);
	gtk_window_set_default_size(GTK_WINDOW(window), width / hidpi_scale, height / hidpi_scale);
	gtk_widget_show_all(window);
	ctx->header_read_tag = 0;
	return FALSE;
}

static int decode_complete(void *arg)
{
	struct img_decode_context *ctx;

	ctx = (struct img_decode_context *)arg;
	ctx->worker_thread = 0;
	if (ctx->sfc_complete) {
		cairo_surface_destroy(ctx->sfc_complete);
	}
	ctx->sfc_complete = ctx->sfc;
	ctx->sfc = NULL;
	cleanup_auto_redraw(ctx);
	cairo_surface_mark_dirty(ctx->sfc_complete);
	gtk_widget_queue_draw(ctx->darea);

	ctx->decode_complete_tag = 0;
	return FALSE;
}

struct worker_thread_cleanups {
	struct jpeg_decompress_struct *dinfo;
	png_structp rpng;
	png_infop rinfo;
	struct oil_libjpeg *olj;
	struct oil_libpng *olp;
	unsigned char *outbuf;
};

struct png_struct_cleanups {
	png_structp rpng;
	png_infop rinfo;
};

struct buffer_desc {
	int width;
	int height;
	int stride;
	unsigned char *buf;
};

void surface_to_write_buffer(cairo_surface_t *sfc, int img_width,
	int img_height, struct buffer_desc *buf_desc)
{
	int buf_width, buf_height, stride, scale_width, scale_height;
	unsigned char *tmp;

	stride = cairo_image_surface_get_stride(sfc);
	buf_width = cairo_image_surface_get_width(sfc);
	buf_height = cairo_image_surface_get_height(sfc);
	tmp = cairo_image_surface_get_data(sfc);

	scale_width = buf_width;
	scale_height = buf_height;
	fix_ratio_shrink_only(img_width, img_height, &scale_width, &scale_height);

	tmp += (stride / 4 - scale_width) / 2 * 4;
	tmp += (buf_height - scale_height) / 2 * stride;

	buf_desc->width = scale_width;
	buf_desc->height = scale_height;
	buf_desc->stride = stride;
	buf_desc->buf = tmp;
}

static void png_cleanup_read_struct(void *arg)
{
	struct png_struct_cleanups *cleanup;
	cleanup = (struct png_struct_cleanups *)arg;
	png_destroy_read_struct(&cleanup->rpng, &cleanup->rinfo, NULL);
}

static void png_cleanup_oil_scale(void *arg)
{
	oil_libpng_free((struct oil_libpng *)arg);
}

static void thread_cleanup_free_buf(void *arg)
{
	free(arg);
}

static void invoke_header_read(struct img_decode_context *ctx)
{
	if (ctx->sfc) {
		return;
	}
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	ctx->header_read_tag = g_idle_add(header_read, ctx);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	sem_wait(&ctx->surface_set_sem);
}

static void png_worker_thread(struct img_decode_context *ctx, FILE *io)
{
	unsigned char *outbuf, *tmp;
	int i, in_width, in_height;
	png_structp rpng;
	png_infop rinfo;
	struct oil_libpng ol;
	struct png_struct_cleanups struct_cleanup;
	struct buffer_desc buf_desc;

	rpng = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

	if (setjmp(png_jmpbuf(rpng))) {
		return;
	}

	rinfo = png_create_info_struct(rpng);
	struct_cleanup.rpng = rpng;
	struct_cleanup.rinfo = rinfo;
	pthread_cleanup_push(png_cleanup_read_struct, (void *)&struct_cleanup);

	png_init_io(rpng, io);
	png_read_info(rpng, rinfo);

	png_set_packing(rpng);
	png_set_strip_16(rpng);
	png_set_expand(rpng);
	png_set_interlace_handling(rpng);
	png_read_update_info(rpng, rinfo);

	in_width = png_get_image_width(rpng, rinfo);
	in_height = png_get_image_height(rpng, rinfo);

	ctx->width = in_width;
	ctx->height = in_height;

	invoke_header_read(ctx);

	surface_to_write_buffer(ctx->sfc, in_width, in_height, &buf_desc);

	oil_libpng_init(&ol, rpng, rinfo, buf_desc.width, buf_desc.height);
	pthread_cleanup_push(png_cleanup_oil_scale, (void *)&ol);

	outbuf = malloc(buf_desc.width * OIL_CMP(ol.os.cs));
	pthread_cleanup_push(thread_cleanup_free_buf, (void *)outbuf);

	for(i=0; i<buf_desc.height; i++) {
		oil_libpng_read_scanline(&ol, outbuf);
		tmp = buf_desc.buf + i * buf_desc.stride;
		translate(outbuf, tmp, buf_desc.width, ol.os.cs);
	}

	pthread_cleanup_pop(1);
	pthread_cleanup_pop(1);
	pthread_cleanup_pop(1);
}

static void jpg_loop_scanlines_translate(struct buffer_desc *desc,
	struct oil_libjpeg *ol)
{
	int i;
	unsigned char *tmp, *outbuf;

	outbuf = malloc(desc->width * OIL_CMP(ol->os.cs));
	pthread_cleanup_push(thread_cleanup_free_buf, (void *)outbuf);

	for(i=0; i<desc->height; i++) {
		tmp = desc->buf + i * desc->stride;
		oil_libjpeg_read_scanline(ol, outbuf);
		translate(outbuf, tmp, desc->width, ol->os.cs);
	}

	pthread_cleanup_pop(1);
}

static void jpg_free(void *arg)
{
	struct jpeg_decompress_struct *dinfo;
	dinfo = (struct jpeg_decompress_struct *)arg;
	jpeg_destroy_decompress(dinfo);
}

static void jpg_cleanup_oil_scale(void *arg)
{
	oil_libjpeg_free((struct oil_libjpeg *)arg);
}

static void jpg_decode(struct jpeg_decompress_struct *dinfo,
	struct buffer_desc *desc)
{
	struct oil_libjpeg ol;
	jpeg_start_decompress(dinfo);
	pthread_cleanup_push(jpg_cleanup_oil_scale, (void *)&ol);
	oil_libjpeg_init(&ol, dinfo, desc->width, desc->height);
	jpg_loop_scanlines_translate(desc, &ol);
	pthread_cleanup_pop(1);
}

static void jpg_worker_thread(struct img_decode_context *ctx, FILE *io)
{
	struct jpeg_decompress_struct dinfo;
	struct jpeg_error_mgr jerr;
	struct buffer_desc buf_desc;

	dinfo.err = jpeg_std_error(&jerr);
	pthread_cleanup_push(jpg_free, (void *)&dinfo);
	jpeg_create_decompress(&dinfo);

	jpeg_stdio_src(&dinfo, io);
	jpeg_read_header(&dinfo, TRUE);
	jpeg_calc_output_dimensions(&dinfo);

	ctx->width = dinfo.output_width;
	ctx->height = dinfo.output_height;
	invoke_header_read(ctx);
	surface_to_write_buffer(ctx->sfc, dinfo.output_width,
		dinfo.output_height, &buf_desc);
	jpg_decode(&dinfo, &buf_desc);
	pthread_cleanup_pop(1);
}

int looks_like_png(FILE *io)
{
	int peek;
	peek = getc(io);
	ungetc(peek, io);
	return peek == 137;
}

static void worker_thread_cleanup_io(void *arg)
{
	fclose((FILE *)arg);
}

/**
 * Open file at ctx->path, peek at the first byte, and pass the io off to
 * the appropriate decoder.
 */
static void *worker_thread(void *arg)
{
	struct img_decode_context *ctx;
	FILE *io;

	ctx = (struct img_decode_context *)arg;
	io = fopen(ctx->path, "r");
	pthread_cleanup_push(worker_thread_cleanup_io, (void*)io);

	if (looks_like_png(io)) {
		png_worker_thread(ctx, io);
	} else {
		jpg_worker_thread(ctx, io);
	}

	pthread_cleanup_pop(1);
	pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);
	ctx->decode_complete_tag = g_idle_add(decode_complete, arg);
	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	return NULL;
}

struct img_decode_context *create_img_decode_context()
{
	struct img_decode_context *ctx;
	ctx = calloc(1, sizeof(struct img_decode_context));
	sem_init(&ctx->surface_set_sem, 0, -1);
	return ctx;
}

void spawn_decoder(struct img_decode_context *ctx)
{
	pthread_create(&ctx->worker_thread, NULL, worker_thread, ctx);
}

static void open_file_existing_window(GtkWidget *window, struct img_decode_context *ctx)
{
	ctx->darea = gtk_drawing_area_new();
	g_signal_connect(ctx->darea, "destroy", G_CALLBACK(on_destroy_handler), ctx);
	g_signal_connect(ctx->darea, "draw", G_CALLBACK(on_draw_event), ctx);
	g_signal_connect(ctx->darea, "configure-event", G_CALLBACK(on_configure_event), ctx);
	gtk_container_add(GTK_CONTAINER(window), ctx->darea);
}

static void open_file_new_window(GtkApplication *app, struct img_decode_context *ctx)
{
	GtkWidget *window;
	spawn_decoder(ctx);
	window = create_window(app, ctx);
	open_file_existing_window(window, ctx);
}

void new_file_chosen(GtkWidget *widget, gpointer user_data)
{
	GtkWidget *window;
	char *new_path;
	struct img_decode_context *ctx;

	ctx = (struct img_decode_context *)user_data;
	new_path = gtk_file_chooser_get_filename(GTK_FILE_CHOOSER(widget));
	window = gtk_widget_get_toplevel(widget);

	if (ctx->path) {
		ctx = create_img_decode_context();
		ctx->path = new_path;
		open_file_new_window(gtk_window_get_application(GTK_WINDOW(window)), ctx);
	} else {
		ctx->path = new_path;
		open_file_existing_window(window, ctx);
		gtk_widget_show_all(window);
	}
}

GtkWidget *create_window(GtkApplication *app, struct img_decode_context *ctx)
{
	GtkWidget *window, *title, *open_button;

	window = gtk_application_window_new(app);
	title = gtk_header_bar_new();
	gtk_header_bar_set_show_close_button(GTK_HEADER_BAR(title), TRUE);
	gtk_header_bar_set_title(GTK_HEADER_BAR(title), g_get_application_name());
	gtk_window_set_titlebar(GTK_WINDOW(window), title);

	open_button = gtk_file_chooser_button_new("Select a file", GTK_FILE_CHOOSER_ACTION_OPEN);
	g_signal_connect(open_button, "file-set", G_CALLBACK(new_file_chosen), ctx);
	gtk_header_bar_pack_end(GTK_HEADER_BAR(title), open_button);

	return window;
}

static void activate(GtkApplication *app, gpointer user_data)
{
	GtkWidget *window;
	struct img_decode_context *ctx;
	ctx = create_img_decode_context();
	window = create_window(app, ctx);
	gtk_widget_show_all(window);
}

static void open(GApplication *app, GFile **files, gint n_files,
	const gchar *hint)
{
	int i;
	struct img_decode_context *ctx;

	for (i=0; i<n_files; i++) {
		ctx = create_img_decode_context();
		ctx->path = g_file_get_path(files[i]);
		open_file_new_window(GTK_APPLICATION(app), ctx);
	}
}

int main(int argc, char *argv[])
{
	GtkApplication *app;
	int status;

	app = gtk_application_new("com.github.ender672.oilview",
		G_APPLICATION_HANDLES_OPEN);
	g_signal_connect(app, "activate", G_CALLBACK(activate), NULL);
	g_signal_connect(app, "open", G_CALLBACK(open), NULL);
	status = g_application_run(G_APPLICATION(app), argc, argv);
	g_object_unref(app);

	return status;
}
