/*
 * File:
 *  video_decoder.c
 *
 * Description:
 *  A video decoder that uses avcodec library
 *
 * Author:
 *  Jonghoon Jin // jhjin0@gmail.com
 *  Marko Vitez // marko@vitez.it
 */

#include <luaT.h>
#include <TH/TH.h>

#include <stdio.h>
#include <unistd.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/mathematics.h>
#include <pthread.h>
#include "http.h"
#ifdef DOVIDEOCAP
#include "videocap.h"
#include "videocodec.h"
#endif

#define BYTE2FLOAT 0.003921568f // 1/255

/* video decoder on DMA memory */
int loglevel = 0;
static int stream_idx;
static AVFormatContext *pFormatCtx;
static AVCodecContext *pCodecCtx;
static AVFrame *pFrame_yuv;
static AVFrame *pFrame_intm;
static AVFormatContext *ofmt_ctx;
static char destfile[500], *destext, destformat[100];
static pthread_t rx_tid;
static int rx_active, frame_decoded;
static pthread_mutex_t readmutex = PTHREAD_MUTEX_INITIALIZER;
static int fragmentsize_seconds;
static int reencode_stream;
static uint64_t start_dts;
static int64_t fragmentsize;
static short audiobuf[16384];
static int audiobuflen;
static int savenow_seconds_before, savenow_seconds_after;
static char savenow_path[300];
static char post_url[300], post_username[100], post_password[100], post_device[100];
#define RXFIFOQUEUESIZE 1000
static AVPacket rxfifo[RXFIFOQUEUESIZE];
static int rxfifo_tail, rxfifo_head;
#ifdef DOVIDEOCAP
static void *vcap, *vcodec, *vcap_frame, *vcodec_extradata;
static int vcap_w, vcap_h, vcap_fps, vcodec_writeextradata, vcodec_extradata_size, vcap_nframes;
const int vcodec_gopsize = 12;
#endif

static struct {
	char *data;
	unsigned datalen;
	long timestamp;
} jpeg;

/* yuv420p-to-rgbp lookup table */
static short TB_YUR[256], TB_YUB[256], TB_YUGU[256], TB_YUGV[256], TB_Y[256];
static uint8_t TB_SAT[1024 + 1024 + 256];

/* This function calculates a lookup table for yuv420p-to-rgbp conversion
 * Written by Marko Vitez.
 */
static void video_decoder_yuv420p_rgbp_LUT()
{
	int i;

	/* calculate lookup table for yuv420p */
	for (i = 0; i < 256; i++) {
		TB_YUR[i]  =  459 * (i-128) / 256;
		TB_YUB[i]  =  541 * (i-128) / 256;
		TB_YUGU[i] = -137 * (i-128) / 256;
		TB_YUGV[i] = - 55 * (i-128) / 256;
		TB_Y[i]    = (i-16) * 298 / 256;
	}
	for (i = 0; i < 1024; i++) {
		TB_SAT[i] = 0;
		TB_SAT[i + 1024 + 256] = 255;
	}
	for (i = 0; i < 256; i++)
		TB_SAT[i + 1024] = i;
}

/* This function is a main function for converting color space from yuv420p to planar RGB.
 * It utilizes a lookup table method for fast conversion. Written by Marko Vitez.
 */
static void video_decoder_yuv420p_rgbp(AVFrame * yuv, AVFrame * rgb)
{
	int i, j, U, V, Y, YUR, YUG, YUB;
	int h = yuv->height;
	int w = yuv->width;
	int wy = yuv->linesize[0];
	int wu = yuv->linesize[1];
	int wv = yuv->linesize[2];
	uint8_t *r = rgb->data[0];
	uint8_t *g = rgb->data[1];
	uint8_t *b = rgb->data[2];
	uint8_t *y = yuv->data[0];
	uint8_t *u = yuv->data[1];
	uint8_t *v = yuv->data[2];
	uint8_t *r1, *g1, *b1, *y1;
	w /= 2;
	h /= 2;

	/* convert for R channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + 2*i*wy + 2*j;
			V     = v[j + i * wv];
			YUR   = TB_YUR[V];
			r1    = (uint8_t *) r + 4*w*i + 2*j;

			Y     = TB_Y[y1[0]];
			*r1++ = TB_SAT[Y + YUR + 1024];
			Y     = TB_Y[y1[1]];
			*r1   = TB_SAT[Y + YUR + 1024];
			y1   += wy;
			r1   += 2*w - 1;
			Y     = TB_Y[y1[0]];
			*r1++ = TB_SAT[Y + YUR + 1024];
			Y     = TB_Y[y1[1]];
			*r1   = TB_SAT[Y + YUR + 1024];
		}
	}

	/* convert for G channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + 2*i*wy + 2*j;
			U     = u[j + i * wu];
			V     = v[j + i * wv];
			YUG   = TB_YUGU[U] + TB_YUGV[V];
			g1    = (uint8_t *) g + 4*w*i + 2*j;

			Y     = TB_Y[y1[0]];
			*g1++ = TB_SAT[Y + YUG + 1024];
			Y     = TB_Y[y1[1]];
			*g1   = TB_SAT[Y + YUG + 1024];
			y1   += wy;
			g1   += 2*w - 1;
			Y     = TB_Y[y1[0]];
			*g1++ = TB_SAT[Y + YUG + 1024];
			Y     = TB_Y[y1[1]];
			*g1   = TB_SAT[Y + YUG + 1024];
		}
	}

	/* convert for B channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + 2*i*wy + 2*j;
			U     = u[j + i * wu];
			YUB   = TB_YUB[U];
			b1    = (uint8_t *) b + 4*w*i + 2*j;

			Y     = TB_Y[y1[0]];
			*b1++ = TB_SAT[Y + YUB + 1024];
			Y     = TB_Y[y1[1]];
			*b1   = TB_SAT[Y + YUB + 1024];
			y1   += wy;
			b1   += 2*w - 1;
			Y     = TB_Y[y1[0]];
			*b1++ = TB_SAT[Y + YUB + 1024];
			Y     = TB_Y[y1[1]];
			*b1   = TB_SAT[Y + YUB + 1024];
		}
	}
}

/* This function is a main function for converting color space from yuv420p to planar RGB
 * directly in torch float tensor
 * It utilizes a lookup table method for fast conversion. Written by Marko Vitez.
 */
static void yuv420p_floatrgbp(AVFrame * yuv, float *dst_float, int imgstride, int rowstride, int w, int h)
{
	int i, j, U, V, Y, YUR, YUG, YUB;
	int wy = yuv->linesize[0];
	int wu = yuv->linesize[1];
	int wv = yuv->linesize[2];
	float *r = dst_float;
	float *g = dst_float + imgstride;
	float *b = dst_float + 2*imgstride;
	uint8_t *y = yuv->data[0];
	uint8_t *u = yuv->data[1];
	uint8_t *v = yuv->data[2];
	uint8_t *y1;
	float *r1, *g1, *b1;
	w /= 2;
	h /= 2;

	/* convert for R channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + 2*i*wy + 2*j;
			V     = v[j + i * wv];
			YUR   = TB_YUR[V];
			r1    = r + 2*(rowstride*i + j);

			Y     = TB_Y[y1[0]];
			*r1++ = TB_SAT[Y + YUR + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*r1   = TB_SAT[Y + YUR + 1024] * BYTE2FLOAT;
			y1   += wy;
			r1   += 2*w - 1;
			Y     = TB_Y[y1[0]];
			*r1++ = TB_SAT[Y + YUR + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*r1   = TB_SAT[Y + YUR + 1024] * BYTE2FLOAT;
		}
	}

	/* convert for G channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + 2*i*wy + 2*j;
			U     = u[j + i * wu];
			V     = v[j + i * wv];
			YUG   = TB_YUGU[U] + TB_YUGV[V];
			g1    = g + 2*(rowstride*i + j);

			Y     = TB_Y[y1[0]];
			*g1++ = TB_SAT[Y + YUG + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*g1   = TB_SAT[Y + YUG + 1024] * BYTE2FLOAT;
			y1   += wy;
			g1   += 2*w - 1;
			Y     = TB_Y[y1[0]];
			*g1++ = TB_SAT[Y + YUG + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*g1   = TB_SAT[Y + YUG + 1024] * BYTE2FLOAT;
		}
	}

	/* convert for B channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + 2*i*wy + 2*j;
			U     = u[j + i * wu];
			YUB   = TB_YUB[U];
			b1    = b + 2*(rowstride*i + j);

			Y     = TB_Y[y1[0]];
			*b1++ = TB_SAT[Y + YUB + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*b1   = TB_SAT[Y + YUB + 1024] * BYTE2FLOAT;
			y1   += wy;
			b1   += 2*w - 1;
			Y     = TB_Y[y1[0]];
			*b1++ = TB_SAT[Y + YUB + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*b1   = TB_SAT[Y + YUB + 1024] * BYTE2FLOAT;
		}
	}
}

/* This function is a main function for converting color space from yuv422p to planar RGB.
 * It utilizes a lookup table method for fast conversion. Written by Marko Vitez.
 */
static void video_decoder_yuv422p_rgbp(AVFrame * yuv, AVFrame * rgb)
{
	int i, j, U, V, Y, YUR, YUG, YUB;
	int h = yuv->height;
	int w = yuv->width;
	int wy = yuv->linesize[0];
	int wu = yuv->linesize[1];
	int wv = yuv->linesize[2];
	uint8_t *r = rgb->data[0];
	uint8_t *g = rgb->data[1];
	uint8_t *b = rgb->data[2];
	uint8_t *y = yuv->data[0];
	uint8_t *u = yuv->data[1];
	uint8_t *v = yuv->data[2];
	uint8_t *r1, *g1, *b1, *y1;
	w /= 2;

	/* convert for R channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + i*wy + 2*j;
			V     = v[j + i * wv];
			YUR   = TB_YUR[V];
			r1    = (uint8_t *) r + 2*(w*i + j);

			Y     = TB_Y[y1[0]];
			*r1++ = TB_SAT[Y + YUR + 1024];
			Y     = TB_Y[y1[1]];
			*r1   = TB_SAT[Y + YUR + 1024];
		}
	}

	/* convert for G channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + i*wy + 2*j;
			U     = u[j + i * wu];
			V     = v[j + i * wv];
			YUG   = TB_YUGU[U] + TB_YUGV[V];
			g1    = (uint8_t *) g + 2*(w*i + j);

			Y     = TB_Y[y1[0]];
			*g1++ = TB_SAT[Y + YUG + 1024];
			Y     = TB_Y[y1[1]];
			*g1   = TB_SAT[Y + YUG + 1024];
		}
	}

	/* convert for B channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + i*wy + 2*j;
			U     = u[j + i * wu];
			YUB   = TB_YUB[U];
			b1    = (uint8_t *) b + 2*(w*i + j);

			Y     = TB_Y[y1[0]];
			*b1++ = TB_SAT[Y + YUB + 1024];
			Y     = TB_Y[y1[1]];
			*b1   = TB_SAT[Y + YUB + 1024];
		}
	}
}

/* This function is a main function for converting color space from yuv422p to planar RGB
 * directly in torch float tensor
 * It utilizes a lookup table method for fast conversion. Written by Marko Vitez.
 */
static void yuv422p_floatrgbp(AVFrame * yuv, float *dst_float, int imgstride, int rowstride, int w, int h)
{
	int i, j, U, V, Y, YUR, YUG, YUB;
	int wy = yuv->linesize[0];
	int wu = yuv->linesize[1];
	int wv = yuv->linesize[2];
	float *r = dst_float;
	float *g = dst_float + imgstride;
	float *b = dst_float + 2*imgstride;
	uint8_t *y = yuv->data[0];
	uint8_t *u = yuv->data[1];
	uint8_t *v = yuv->data[2];
	uint8_t *y1;
	float *r1, *g1, *b1;
	w /= 2;

	/* convert for R channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + i*wy + 2*j;
			V     = v[j + i * wv];
			YUR   = TB_YUR[V];
			r1    = r + i*rowstride + 2*j;

			Y     = TB_Y[y1[0]];
			*r1++ = TB_SAT[Y + YUR + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*r1   = TB_SAT[Y + YUR + 1024] * BYTE2FLOAT;
		}
	}

	/* convert for G channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + i*wy + 2*j;
			U     = u[j + i * wu];
			V     = v[j + i * wv];
			YUG   = TB_YUGU[U] + TB_YUGV[V];
			g1    = g + i*rowstride + 2*j;

			Y     = TB_Y[y1[0]];
			*g1++ = TB_SAT[Y + YUG + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*g1   = TB_SAT[Y + YUG + 1024] * BYTE2FLOAT;
		}
	}

	/* convert for B channel */
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			y1    = y + i*wy + 2*j;
			U     = u[j + i * wu];
			YUB   = TB_YUB[U];
			b1    = b + i*rowstride + 2*j;

			Y     = TB_Y[y1[0]];
			*b1++ = TB_SAT[Y + YUB + 1024] * BYTE2FLOAT;
			Y     = TB_Y[y1[1]];
			*b1   = TB_SAT[Y + YUB + 1024] * BYTE2FLOAT;
		}
	}
}

/* This function is a main function for converting color space from yuv420p to planar YUV.
 * Written by Marko Vitez.
 */
static void video_decoder_yuv420p_yuvp(AVFrame * yuv420, AVFrame * yuv)
{
	int i, j, j2, k;
	int h = yuv420->height;
	int w = yuv420->width;
	int wy = yuv420->linesize[0];
	int wu = yuv420->linesize[1];
	int wv = yuv420->linesize[2];

	uint8_t *srcy = yuv420->data[0];
	uint8_t *srcu = yuv420->data[1];
	uint8_t *srcv = yuv420->data[2];

	uint8_t *dsty = yuv->data[0];
	uint8_t *dstu = yuv->data[1];
	uint8_t *dstv = yuv->data[2];

	uint8_t *dst_y, *dst_u, *dst_v;
	uint8_t *src_y, *src_u, *src_v;

	for (i = 0; i < h; i++) {
		src_y = &srcy[i * wy];
		src_u = &srcu[i * wu / 2];
		src_v = &srcv[i * wv / 2];

		dst_y = &dsty[i * w];
		dst_u = &dstu[i * w];
		dst_v = &dstv[i * w];

		for (j = 0, k = 0; j < w; j++, k += 1) {
			j2 = j >> 1;

			dst_y[k] = src_y[j];
			dst_u[k] = src_u[j2];
			dst_v[k] = src_v[j2];
		}
	}
}

/* This function is a main function for converting color space from YUYV to planar RGB.
 * It can fill directly a torch byte tensor
 * Written by Marko Vitez.
 */
void yuyv2torchRGB(const unsigned char *frame, unsigned char *dst_byte, int imgstride, int rowstride, int w, int h)
{
	int i, j, w2 = w / 2;
	uint8_t *dst;
	const uint8_t *src;

	/* convert for R channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_byte + i * rowstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUR[ src[3] ] + 1024];
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUR[ src[3] ] + 1024];
			src += 4;
		}
	}

	/* convert for G channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_byte + i * rowstride + imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024];
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024];
			src += 4;
		}
	}

	/* convert for B channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_byte + i * rowstride + 2*imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUB[ src[1] ] + 1024];
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUB[ src[1] ] + 1024];
			src += 4;
		}
	}
}

/* This function is a main function for converting color space from YUYV to planar RGB.
 * It can fill directly a torch float tensor
 * Written by Marko Vitez.
 */
void yuyv2torchfloatRGB(const unsigned char *frame, float *dst_float, int imgstride, int rowstride, int w, int h)
{
	int i, j, w2 = w / 2;
	float *dst;
	const uint8_t *src;

	/* convert for R channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_float + i * rowstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUR[ src[3] ] + 1024] * BYTE2FLOAT;
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUR[ src[3] ] + 1024] * BYTE2FLOAT;
			src += 4;
		}
	}

	/* convert for G channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_float + i * rowstride + imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024] * BYTE2FLOAT;
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUGU[ src[1] ] + TB_YUGV[ src[3] ] + 1024] * BYTE2FLOAT;
			src += 4;
		}
	}

	/* convert for B channel */
	src = frame;
	for (i = 0; i < h; i++) {
		dst = dst_float + i * rowstride + 2*imgstride;
		for (j = 0; j < w2; j++) {
			*dst++ = TB_SAT[ TB_Y[ src[0] ] + TB_YUB[ src[1] ] + 1024] * BYTE2FLOAT;
			*dst++ = TB_SAT[ TB_Y[ src[2] ] + TB_YUB[ src[1] ] + 1024] * BYTE2FLOAT;
			src += 4;
		}
	}
}

/*
 * Free and close video decoder
 */
int video_decoder_exit(lua_State * L)
{
	if(rx_tid)
	{
		void *retval;
		rx_active = 0;
		pthread_join(rx_tid, &retval);
		rx_tid = 0;
	}
	
#ifdef DOVIDEOCAP
	if(vcap)
	{
		videocap_close(vcap);
		vcap = 0;
	}
	if(vcodec)
	{
		videocodec_close(vcodec);
		vcodec = 0;
	}
	if(vcap_frame)
	{
		free(vcap_frame);
		vcap_frame = 0;
	}
	if(vcodec_extradata)
	{
		free(vcodec_extradata);
		vcodec_extradata = 0;
	}
#endif
	/* free the AVFrame structures */
	if (pFrame_intm) {
		av_free(pFrame_intm);
		pFrame_intm = 0;
	}
	if (pFrame_yuv) {
		av_free(pFrame_yuv);
		pFrame_yuv = 0;
	}

	/* close the codec and video file */
	if (pCodecCtx) {
		avcodec_close(pCodecCtx);
		pCodecCtx = 0;
	}
	if (pFormatCtx)
		avformat_close_input(&pFormatCtx);
	pFormatCtx = 0;
	frame_decoded = 0;
	mpjpeg_disconnect();
	return 0;
}

/* This function initiates libavcodec and its utilities. It finds a valid stream from
 * the given video file and returns the height and width of the input video. The input
 * arguments is the location of file in a string.
 */
 
static int video_decoder_init(lua_State * L)
{
	int i;
	AVCodec *pCodec;

	video_decoder_exit(NULL);
	/* pass input arguments */
	const char *fpath = lua_tostring(L, 1);
	const char *src_type = lua_tostring(L, 2);
	if(loglevel >= 3)
		fprintf(stderr, "video_decoder_init(%s,%s)\n", fpath, src_type);

	if(src_type && !strcmp(src_type, "MPJPEG"))
	{
		// For JPEGs coming from a webserver with multipart content-type
		// we have our routines
		AVPacket pkt;
		int rc = mpjpeg_connect(fpath);
		if(rc)
			luaL_error(L, "Connection to %s failed: %s", fpath, http_error(rc));
		rc = mpjpeg_getdata(&jpeg.data, &jpeg.datalen, &jpeg.timestamp);
		if(rc)
			luaL_error(L, "Connection to %s failed: %s", fpath, http_error(rc));
		// Create a JPEG decoder
		pCodec = avcodec_find_decoder(CODEC_ID_MJPEG);
		if (pCodec == NULL)
			luaL_error(L, "<video_decoder> the codec is not supported");
		pCodecCtx = avcodec_alloc_context3(pCodec);
		if(!pCodecCtx)
			luaL_error(L, "<video_decoder> error allocating codec");
		if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
			video_decoder_exit(NULL);
			luaL_error(L, "<video_decoder> could not open the codec");
		}
		memset(&pkt, 0, sizeof(pkt));
		av_init_packet(&pkt);
		pkt.data = (unsigned char *)jpeg.data;
		pkt.size = jpeg.datalen;
		pkt.flags = AV_PKT_FLAG_KEY;
		pFrame_yuv = avcodec_alloc_frame();
		if(avcodec_decode_video2(pCodecCtx, pFrame_yuv, &i, &pkt) < 0)
		{
			video_decoder_exit(NULL);
			luaL_error(L, "<video_decoder> Error decoding JPEG image");
		}
		lua_pushboolean(L, 1);
		lua_pushnumber(L, pCodecCtx->height);
		lua_pushnumber(L, pCodecCtx->width);
		lua_pushnil(L);
		lua_pushnil(L);
		return 5;
	}
	/* use the input format if provided, otherwise guess */
	AVInputFormat *iformat = av_find_input_format(src_type);

	/* open video file */
	if (avformat_open_input(&pFormatCtx, fpath, iformat, NULL) != 0) {
		video_decoder_exit(NULL);
		luaL_error(L, "<video_decoder> no video was provided");
	}

	/* retrieve stream information */
	if (avformat_find_stream_info(pFormatCtx, NULL) < 0) {
		video_decoder_exit(NULL);
		luaL_error(L, "<video_decoder> no stream information was found");
	}

	/* dump information about file onto standard error */
	if (loglevel > 0) av_dump_format(pFormatCtx, 0, fpath, 0);

	/* find the first video stream */
	stream_idx = -1;
	for (i = 0; i < pFormatCtx->nb_streams; i++) {
		if (pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			stream_idx = i;
			break;
		}
	}
	if (stream_idx == -1) {
		video_decoder_exit(NULL);
		luaL_error(L, "<video_decoder> could not find a video stream");
	}

	/* get a pointer to the codec context for the video stream */
	pCodecCtx = pFormatCtx->streams[stream_idx]->codec;

	/* find the decoder for the video stream */
	pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
	if (pCodec == NULL) {
		video_decoder_exit(NULL);
		luaL_error(L, "<video_decoder> the codec is not supported");
	}

	/* open codec */
	if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
		video_decoder_exit(NULL);
		luaL_error(L, "<video_decoder> could not open the codec");
	}

	/* allocate a raw AVFrame structure (yuv420p) */
	pFrame_yuv = avcodec_alloc_frame();

	/* allocate an AVFrame structure (No DMA memory) */
	pFrame_intm = avcodec_alloc_frame();
	pFrame_intm->height = pCodecCtx->height;
	pFrame_intm->width = pCodecCtx->width;
	pFrame_intm->data[0] = av_malloc(pCodecCtx->width * pCodecCtx->height);
	pFrame_intm->data[1] = av_malloc(pCodecCtx->width * pCodecCtx->height);
	pFrame_intm->data[2] = av_malloc(pCodecCtx->width * pCodecCtx->height);

    /* calculate fps */
	double frame_rate = pFormatCtx->streams[stream_idx]->avg_frame_rate.num /
		(double) pFormatCtx->streams[stream_idx]->avg_frame_rate.den;

	if(loglevel >= 3)
		fprintf(stderr, "video_decoder_init ok, %dx%d, %ld frames, %f fps\n", pCodecCtx->width,
			pCodecCtx->height, (long)pFormatCtx->streams[i]->nb_frames, frame_rate);
	/* return frame dimensions */
	lua_pushboolean(L, 1);
	lua_pushnumber(L, pCodecCtx->height);
	lua_pushnumber(L, pCodecCtx->width);
	if (pFormatCtx->streams[stream_idx]->nb_frames > 0) {
		lua_pushnumber(L, pFormatCtx->streams[stream_idx]->nb_frames);
	} else {
		lua_pushnil(L);
	}
	if (frame_rate > 0) {
		lua_pushnumber(L, frame_rate);
	} else {
		lua_pushnil(L);
	}

	return 5;
}

/* This function decodes each frame on the fly. Frame is decoded and saved
 * into "pFrame_yuv" as yuv420p, then converted to planar RGB in
 * "pFrame_intm". Finally, memcpy copies all from "pFrame_intm" to
 * dst tensor, which is measured faster than direct writing of planar RGB
 * to dst tensor.
 * This function has been updated to consider two new situations:
 * 1) If vcap!=0, capture has been started by videocap_init, in this case it takes
 *    the frames from the videocap library instead of using libav; videocap support
 *    has to be explicitly enabled by defining DOVIDEOCAP, because it's only present
 *    on Linux; it's supposed that the V4L2 device outputs frames in the YUVV format;
 *    not every webcam supports this format, but it's very common
 * 2) If rx_tid!=0, decoding occurs in another thread (started by startremux), so this
 *    routine only returns the last decoded frame
 */
static int video_decoder_rgb(lua_State * L)
{
	AVPacket packet;
	int c;
	int dim = 0;
	long *stride = NULL;
	long *size = NULL;
	unsigned char *dst_byte = NULL;
	float *dst_float = NULL;

	const char *tname = luaT_typename(L, 1);
	if (strcmp("torch.ByteTensor", tname) == 0) {
		THByteTensor *frame =
		    luaT_toudata(L, 1, luaT_typenameid(L, "torch.ByteTensor"));

		// get tensor's Info
		dst_byte = THByteTensor_data(frame);
		dim = frame->nDimension;
		stride = &frame->stride[0];
		size = &frame->size[0];

	} else if (strcmp("torch.FloatTensor", tname) == 0) {
		THFloatTensor *frame =
		    luaT_toudata(L, 1, luaT_typenameid(L, "torch.FloatTensor"));

		// get tensor's Info
		dst_float = THFloatTensor_data(frame);
		dim = frame->nDimension;
		stride = &frame->stride[0];
		size = &frame->size[0];

	} else {
		luaL_error(L, "<video_decoder>: cannot process tensor type %s", tname);
	}

	if ((3 != dim) || (3 != size[0])) {
		luaL_error(L, "<video_decoder>: cannot process tensor of this dimension and size");
	}
	
#ifdef DOVIDEOCAP
	if(vcap)
	{
		if(rx_tid)
		{
			// Wait for the first frame to be decoded
			if(!frame_decoded)
			{
				while(rx_tid && !frame_decoded)
					usleep(10000);
			}
			// pFrame_yuv should not be read while it's being written, so lock a mutex
			pthread_mutex_lock(&readmutex);
			if(!frame_decoded)
			{
				pthread_mutex_unlock(&readmutex);
				lua_pushboolean(L, 0);
				return 1;
			}
			// Convert image from YUYV to RGB torch tensor
			if(dst_byte)
				yuyv2torchRGB((unsigned char *)vcap_frame, dst_byte, stride[0], stride[1], vcap_w, vcap_h);
			else yuyv2torchfloatRGB((unsigned char *)vcap_frame, dst_float, stride[0], stride[1], vcap_w, vcap_h);
			pthread_mutex_unlock(&readmutex);
			lua_pushboolean(L, 1);
			return 1;
		}
		char *frame;
		struct timeval tv;

		// Get the frame from the V4L2 device using our videocap library
		int rc = videocap_getframe(vcap, &frame, &tv);
		if(rc < 0)
		{
			luaL_error(L, "videocap_getframe returned error %d", rc);
		}
		// Convert image from YUYV to RGB torch tensor
		if(dst_byte)
			yuyv2torchRGB((unsigned char *)frame, dst_byte, stride[0], stride[1], vcap_w, vcap_h);
		else yuyv2torchfloatRGB((unsigned char *)frame, dst_float, stride[0], stride[1], vcap_w, vcap_h);
		lua_pushboolean(L, 1);
		return 1;
	}
#endif
	if(rx_tid)
	{
		// Wait for the first frame to be decoded
		if(!frame_decoded)
		{
			while(rx_tid && !frame_decoded)
				usleep(10000);
		}
		// pFrame_yuv should not be read while it's being written, so lock a mutex
		pthread_mutex_lock(&readmutex);
		if(!frame_decoded)
		{
			pthread_mutex_unlock(&readmutex);
			lua_pushboolean(L, 0);
			return 1;
		}
		// Convert from YUV to RGB
		if(dst_byte)
		{
			if(pCodecCtx->pix_fmt == PIX_FMT_YUV422P || pCodecCtx->pix_fmt == PIX_FMT_YUVJ422P)
				video_decoder_yuv422p_rgbp(pFrame_yuv, pFrame_intm);
			else video_decoder_yuv420p_rgbp(pFrame_yuv, pFrame_intm);
			pthread_mutex_unlock(&readmutex);

			/* copy each channel from av_malloc to DMA_malloc */
			for (c = 0; c < dim; c++)
				memcpy(dst_byte + c * stride[0],
					   pFrame_intm->data[c],
					   size[1] * size[2]);
		} else {
			if(pCodecCtx->pix_fmt == PIX_FMT_YUV422P || pCodecCtx->pix_fmt == PIX_FMT_YUVJ422P)
				yuv422p_floatrgbp(pFrame_yuv, dst_float, stride[0], stride[1], pCodecCtx->width, pCodecCtx->height);
			else yuv420p_floatrgbp(pFrame_yuv, dst_float, stride[0], stride[1], pCodecCtx->width, pCodecCtx->height);
			pthread_mutex_unlock(&readmutex);
		}

		lua_pushboolean(L, 1);
		return 1;
	}
	while ( (pFormatCtx && av_read_frame(pFormatCtx, &packet) >= 0) ||
		(!pFormatCtx && !mpjpeg_getdata(&jpeg.data, &jpeg.datalen, &jpeg.timestamp)) )
	{
		if(!pFormatCtx)
		{
			// We are getting data from mpjpeg here, not avformat
			memset(&packet, 0, sizeof(packet));
			av_init_packet(&packet);
			packet.data = (unsigned char *)jpeg.data;
			packet.size = jpeg.datalen;
			packet.flags = AV_PKT_FLAG_KEY;
			packet.stream_index = stream_idx;
			pFrame_yuv = avcodec_alloc_frame();		
		}
		/* is this a packet from the video stream? */
		if (packet.stream_index == stream_idx) {

			/* decode video frame */
			avcodec_decode_video2(pCodecCtx, pFrame_yuv, &frame_decoded, &packet);

			/* check if frame is decoded */
			if (frame_decoded) {

				/* convert YUV420p to planar RGB */
				if(dst_byte)
				{
					if(pCodecCtx->pix_fmt == PIX_FMT_YUV422P || pCodecCtx->pix_fmt == PIX_FMT_YUVJ422P)
						video_decoder_yuv422p_rgbp(pFrame_yuv, pFrame_intm);
					else video_decoder_yuv420p_rgbp(pFrame_yuv, pFrame_intm);

					/* copy each channel from av_malloc to DMA_malloc */
					for (c = 0; c < dim; c++)
						memcpy(dst_byte + c * stride[0],
							   pFrame_intm->data[c],
							   size[1] * size[2]);
				} else {
					if(pCodecCtx->pix_fmt == PIX_FMT_YUV422P || pCodecCtx->pix_fmt == PIX_FMT_YUVJ422P)
						yuv422p_floatrgbp(pFrame_yuv, dst_float, stride[0], stride[1], pCodecCtx->width, pCodecCtx->height);
					else yuv420p_floatrgbp(pFrame_yuv, dst_float, stride[0], stride[1], pCodecCtx->width, pCodecCtx->height);
				}

				av_free_packet(&packet);
				lua_pushboolean(L, 1);
				if(!pFormatCtx)
				{
					lua_pushinteger(L, jpeg.timestamp);
					return 2;
				}
				return 1;
			}
		}
		/* free the packet that was allocated by av_read_frame */
		av_free_packet(&packet);
	}

	lua_pushboolean(L, 0);
	return 1;
}

// This routine only supports regular libav frames, no vcap, no startremux thread

static int video_decoder_yuv(lua_State * L)
{
	AVPacket packet;
	int c;
	int dim = 0;
	long *stride = NULL;
	long *size = NULL;
	unsigned char *dst_byte = NULL;

	const char *tname = luaT_typename(L, 1);
	if (strcmp("torch.ByteTensor", tname) == 0) {
		THByteTensor *frame =
		    luaT_toudata(L, 1, luaT_typenameid(L, "torch.ByteTensor"));

		// get tensor's Info
		dst_byte = THByteTensor_data(frame);
		dim = frame->nDimension;
		stride = &frame->stride[0];
		size = &frame->size[0];

	} else {
		luaL_error(L, "<video_decoder>: cannot process tensor type %s", tname);
	}

	if ((3 != dim) || (3 != size[0])) {
		luaL_error(L, "<video_decoder>: cannot process tensor of this dimension and size");
	}

	/* read frames and save first five frames to disk */
	while (av_read_frame(pFormatCtx, &packet) >= 0) {

		/* is this a packet from the video stream? */
		if (packet.stream_index == stream_idx) {

			/* decode video frame */
			avcodec_decode_video2(pCodecCtx, pFrame_yuv, &frame_decoded, &packet);

			/* check if frame is decoded */
			if (frame_decoded) {

				/* convert YUV420p to planar YUV */
				video_decoder_yuv420p_yuvp(pFrame_yuv, pFrame_intm);

				/* copy each channel from av_malloc to DMA_malloc */
				for (c = 0; c < dim; c++)
					memcpy(dst_byte + c * stride[0],
					       pFrame_intm->data[c],
					       size[1] * size[2]);

				av_free_packet(&packet);
				lua_pushboolean(L, 1);
				return 1;
			}
		}
		/* free the packet that was allocated by av_read_frame */
		av_free_packet(&packet);
	}

	lua_pushboolean(L, 0);
	return 1;
}

static void log_packet(const AVFormatContext *fmt_ctx, const AVPacket *pkt, const char *tag)
{
	if(loglevel < 7)
		return;
	if(!fmt_ctx)
		fprintf(stderr, "%s stream=%d dur=%d dts=%ld pts=%ld len=%d %s\n", tag, pkt->stream_index, pkt->duration,
			(long)pkt->dts,
			(long)pkt->pts,
			pkt->size, pkt->flags & AV_PKT_FLAG_KEY ? "KEY" : "");
	else fprintf(stderr, "%s stream=%d dur=%d dts=%f pts=%f len=%d %s\n", tag, pkt->stream_index, pkt->duration,
			(double)pkt->dts * fmt_ctx->streams[pkt->stream_index]->time_base.num / fmt_ctx->streams[pkt->stream_index]->time_base.den,
			(double)pkt->pts * fmt_ctx->streams[pkt->stream_index]->time_base.num / fmt_ctx->streams[pkt->stream_index]->time_base.den,
			pkt->size, pkt->flags & AV_PKT_FLAG_KEY ? "KEY" : "");
}

// Open an AVFormatContext for output to destpath with optional format destformat
// Copy most parameters from the already opened input AVFormatContext
static AVFormatContext *openoutput(lua_State *L, const char *destformat, const char *path)
{
	AVFormatContext *ofmt_ctx;
	int i, ret;
	struct tm tm;
	time_t t;
	char destpath[300];
	char s[300];			
	
	time(&t);
	tm = *localtime(&t);
	if(path)
		strcpy(destpath, path);
	else if(fragmentsize_seconds != -1)
		sprintf(destpath, "%s_%04d%02d%02d-%02d%02d%02d.%s", destfile, 1900 + tm.tm_year, tm.tm_mon+1, tm.tm_mday,
			tm.tm_hour, tm.tm_min, tm.tm_sec, destext);
	else strcpy(destpath, destfile);
	reencode_stream = -1;
	ofmt_ctx = avformat_alloc_context();
	if(!ofmt_ctx)
	{
		if(L)
			luaL_error(L, "Error allocating format context");
		else fprintf(stderr, "Error allocating format context\n");
		return 0;
	}
	ofmt_ctx->oformat = av_guess_format(destformat, 0, 0);
	if(!ofmt_ctx->oformat)
	{
		avformat_free_context(ofmt_ctx);
		if(L)
			luaL_error(L, "Unrecognixed output format %s", destformat);
		else fprintf(stderr, "Unrecognixed output format %s\n", destformat);
		return 0;
	}
	ofmt_ctx->priv_data = NULL;
	// Open the output file
	strncpy(ofmt_ctx->filename, destpath, sizeof(ofmt_ctx->filename));
	if(avio_open(&ofmt_ctx->pb, destpath, AVIO_FLAG_WRITE) < 0)
	{
		if(L)
			luaL_error(L, "Error creating output file %s", destpath);
		else fprintf(stderr, "Error creating output file %s\n", destpath);
		avformat_free_context(ofmt_ctx);
		return 0;
	}
	if(pFormatCtx)
	{
		// Copy the stream contexts
		for (i = 0; i < pFormatCtx->nb_streams; i++) {
			AVStream *in_stream, *out_stream;
			
			in_stream = pFormatCtx->streams[i];
			// If we output in the mp4 container, it's supposed that we will put
			// audio in AAC, so force it
			if(in_stream->codec->codec_type == AVMEDIA_TYPE_AUDIO &&
				in_stream->codec->codec_id != CODEC_ID_AAC
				&& !strcmp(ofmt_ctx->oformat->name, "mp4"))
			{
				AVCodec *decoder;
				
				out_stream = avformat_new_stream(ofmt_ctx, avcodec_find_encoder(CODEC_ID_AAC));
				out_stream->codec->sample_rate = in_stream->codec->sample_rate;
				out_stream->codec->channels = in_stream->codec->channels;
				out_stream->codec->codec_id = CODEC_ID_AAC;
				out_stream->codec->codec_type = AVMEDIA_TYPE_AUDIO;
				out_stream->codec->frame_size = 1024;
				out_stream->codec->sample_fmt = in_stream->codec->sample_fmt;
				reencode_stream  = i;
				decoder = avcodec_find_decoder(in_stream->codec->codec_id);
				if(decoder == NULL)
				{
					if(L)
						luaL_error(L, "Failed to find audio decoder");
					else fprintf(stderr, "Failed to find audio decoder\n");
					avformat_free_context(ofmt_ctx);
					return 0;
				}
				if(avcodec_open2(in_stream->codec, decoder, 0) < 0)
				{
					if(L)
						luaL_error(L, "Failed to open audio decoder");
					else fprintf(stderr, "Failed to open audio decoder\n");
					avformat_free_context(ofmt_ctx);
					return 0;
				}
				// Required to create the proper stream for the MP4 container
				out_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
				if((ret = avcodec_open2(out_stream->codec, 0, 0)) < 0)
				{
					avcodec_close(in_stream->codec);
					av_strerror(ret, s, sizeof(s));
					if(L)
						luaL_error(L, "Failed to open audio encoder: %s", s);
					else fprintf(stderr, "Failed to open audio encoder: %s\n", s);
					avformat_free_context(ofmt_ctx);
					return 0;
				}
			} else {
				out_stream = avformat_new_stream(ofmt_ctx, (AVCodec *)in_stream->codec->codec);
				if (!out_stream) {
					if(L)
						luaL_error(L, "Failed allocating output stream");
					else fprintf(stderr, "Failed allocating output stream\n");
					avformat_free_context(ofmt_ctx);
					return 0;
				}
				ret = avcodec_copy_context(out_stream->codec, in_stream->codec);
				if (ret < 0) {
					if(L)
						luaL_error(L, "Failed to copy context from input to output stream codec context");
					else fprintf(stderr, "Failed to copy context from input to output stream codec context\n");
					avformat_free_context(ofmt_ctx);
					return 0;
				}
				// For MJPEG the aspect ratio is 0/0, fix it
				if(!strcmp(destformat, "avi"))
				{
					out_stream->codec->sample_aspect_ratio.num = 1;
					out_stream->codec->sample_aspect_ratio.den = 1;
				}
				// Copy the aspect ration from the codec to the stream
				out_stream->sample_aspect_ratio = out_stream->codec->sample_aspect_ratio;
				// Take the default codec tag
				out_stream->codec->codec_tag = 0;
			}
			// Use the default time base of 1/90000 seconds used for MPEG-2 TS, it should be good also for other formats
			out_stream->codec->time_base.num = 1;
			out_stream->codec->time_base.den = 90000;
			// If the output fomat requires global hreader, set it
			if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
					out_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
		}
		// Write the header for the container
		ret = avformat_write_header(ofmt_ctx, NULL);
		if (ret < 0) {
			av_strerror(ret, s, sizeof(s));
			if(L)
				luaL_error(L, "Error writing header: %s", s);
			else fprintf(stderr, "Error writing header: %s\n", s);
			avformat_free_context(ofmt_ctx);
			return 0;
		}
	}
#ifdef DOVIDEOCAP
	else {
		// vcap case, create one video stream
		AVCodec *codec;
		AVStream *stream;
		
		vcodec_writeextradata = 1;
		if(destformat)
		{
			if(!strcmp(destformat, "mp4"))
				vcodec_writeextradata = 0;
		}
		codec = avcodec_find_decoder(CODEC_ID_H264);
		stream = avformat_new_stream(ofmt_ctx, codec);
		if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
			stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
		stream->codec->width = vcap_w;
		stream->codec->height = vcap_h;
		stream->codec->coded_width = vcap_w;
		stream->codec->coded_height = vcap_h;
		stream->codec->time_base.num = 1;
		stream->codec->time_base.den = 90000;
		stream->codec->ticks_per_frame = 2;
		stream->codec->pix_fmt = 0;
		stream->time_base.num = 1;
		stream->time_base.den = 90000;
		stream->avg_frame_rate.num = vcap_fps;
		stream->avg_frame_rate.den = 1;
		stream->codec->codec_id = CODEC_ID_H264;
		stream->codec->sample_aspect_ratio.num = 1;
		stream->codec->sample_aspect_ratio.den = 1;
		stream->sample_aspect_ratio = stream->codec->sample_aspect_ratio;
		stream->codec->extradata = av_malloc(vcodec_extradata_size);
		stream->codec->extradata_size = vcodec_extradata_size;
		memcpy(stream->codec->extradata, vcodec_extradata, vcodec_extradata_size);
		// Write the header for the container
		ret = avformat_write_header(ofmt_ctx, NULL);
		if (ret < 0) {
			av_strerror(ret, s, sizeof(s));
			if(L)
				luaL_error(L, "Error writing header: %s", s);
			else fprintf(stderr, "Error writing header: %s\n", s);
			return 0;
		}
    }
#endif
	return ofmt_ctx;
}

void *rx_postfile_thread(void *path1)
{
	char *path = (char *)path1, *destfilename;
	char retbuf[3000];
	int rc;
	
	destfilename = strrchr(path, '/');
	if(destfilename)
		destfilename++;
	else destfilename = path;
	if(loglevel >= 2)
		fprintf(stderr, "Uploading [%s] to [%s] as [%s] with username=[%s] password=[%s] device=[%s]\n", path, post_url,
			destfilename, post_username, post_password, post_device);
	rc = postfile(post_url, path, destfilename, post_username, post_password, post_device, retbuf, sizeof(retbuf));
	if(loglevel >= 2)
		fprintf(stderr, "postfile %s rc=%d (%s)\n%s\n", path, rc, http_error(rc), retbuf);
	unlink(path);
	free(path);
	return 0;
}

void rx_postfile(const char *path)
{
	pthread_t tid;
	pthread_create(&tid, 0, rx_postfile_thread, strdup(path));
	pthread_detach(tid);
}

// Write the input packet pkt to the output stream
static int write_packet(struct AVPacket *pkt, AVRational time_base)
{
	AVStream *out_stream;
	int ret = 0;
	char s[300];
	uint64_t ss;	// start_stream to subtract

	// Calculate the packet parameters for the output stream
	if(pkt->dts == AV_NOPTS_VALUE)
		pkt->dts = 0;
	if(start_dts == -1)
		start_dts = pkt->dts;
	if(pkt->stream_index != reencode_stream)
	{
		if(loglevel >= 5)
			fprintf(stderr, "Write dts=%ld start=%ld size=%ld\n", (long)pkt->dts, (long)start_dts, (long)fragmentsize);
		// If the desired fragment size has been reached and the frame is a key (intra) frame,
		// create a new fragment (we want each fragment to start with a key frame, since
		// inter (non-intra) frames cannot be decoded without a starting key frame
		if(fragmentsize && fragmentsize != -1 &&
			pkt->stream_index == stream_idx && pkt->dts > start_dts + fragmentsize && pkt->flags & AV_PKT_FLAG_KEY)
		{
			if(loglevel >= 4)
				fprintf(stderr, "Close (dts=%ld start=%ld size=%ld)\n", (long)pkt->dts, (long)start_dts, (long)fragmentsize);
			// Write the trailer and close the file
			av_write_trailer(ofmt_ctx);

			/* close output */
			if (ofmt_ctx && !(ofmt_ctx->flags & AVFMT_NOFILE))
			{
				avio_close(ofmt_ctx->pb);
				if(*post_url)
					rx_postfile(ofmt_ctx->filename);
			}
			avformat_free_context(ofmt_ctx);
			ofmt_ctx = 0;
			
			// Only if we are continuously creating files, create the new file
			if(fragmentsize_seconds)
			{
				// Open the new file
				ofmt_ctx = openoutput(0, destformat, 0);
				if(!ofmt_ctx)
					return 0;
				start_dts = pkt->dts;
			} else fragmentsize = 0;
		}
		if(ofmt_ctx)
		{
			// Change timing to output stream requirements
			out_stream = ofmt_ctx->streams[pkt->stream_index];
			pkt->duration = av_rescale_q(pkt->duration, time_base, out_stream->time_base);
			if(pFormatCtx)
				ss = av_rescale_q(start_dts, pFormatCtx->streams[stream_idx]->time_base, time_base);
			else ss = start_dts;
			pkt->pts = av_rescale_q(pkt->pts - ss, time_base, out_stream->time_base);
			pkt->dts = av_rescale_q(pkt->dts - ss, time_base, out_stream->time_base);
			pkt->pos = -1;
#ifdef DOVIDEOCAP
			if(vcodec_writeextradata && (pkt->flags & AV_PKT_FLAG_KEY))
			{
				// Streaming formats need extradata to be written periodically
				AVPacket pkt2;
			
				pkt2 = *pkt;
				pkt->dts++;
				pkt->pts++;
				pkt2.data = vcodec_extradata;
				pkt2.size = vcodec_extradata_size;
				log_packet(ofmt_ctx, &pkt2, "extra");
				ret = av_write_frame(ofmt_ctx, &pkt2);
				if (ret < 0) {
					av_strerror(ret, s, sizeof(s));
					fprintf(stderr, "Error muxing packet: %s\n", s);
				}
			}
#endif
			log_packet(ofmt_ctx, pkt, "out");
			ret = av_write_frame(ofmt_ctx, pkt);				
			if (ret < 0) {
				av_strerror(ret, s, sizeof(s));
				fprintf(stderr, "Error muxing packet: %s\n", s);
			}
		}
	} else {
		AVFrame frame;
		int got;
		
		memset(&frame, 0, sizeof(frame));
		if(avcodec_decode_audio4(pFormatCtx->streams[reencode_stream]->codec, &frame, &got, pkt) >= 0 && got)
		{
			AVPacket pkt2;
			int rc;
			
			memcpy(audiobuf + audiobuflen, frame.data[0], frame.nb_samples * 2);
			audiobuflen += frame.nb_samples;
			memset(&pkt2, 0, sizeof(pkt2));
			if(audiobuflen >= 1024)
			{
				frame.data[0] = (uint8_t *)audiobuf;
				frame.nb_samples = 1024;
				rc = avcodec_encode_audio2(ofmt_ctx->streams[reencode_stream]->codec, &pkt2, &frame, &got);
				if(!rc && got)
				{
					// Change timing to output stream requirements
					out_stream = ofmt_ctx->streams[pkt->stream_index];
					ss = av_rescale_q(start_dts, pFormatCtx->streams[stream_idx]->time_base, time_base);
					pkt2.dts = pkt2.pts = av_rescale_q(pkt->dts - ss, time_base, out_stream->time_base);
					pkt2.duration = 1024 * 90000 / 8000;
					pkt2.stream_index = pkt->stream_index;
					log_packet(ofmt_ctx, &pkt2, "out");
					// Write the packet
					ret = av_write_frame(ofmt_ctx, &pkt2);
					if (ret < 0) {
						av_strerror(ret, s, sizeof(s));
						fprintf(stderr, "Error muxing packet: %s\n", s);
					}
					av_free_packet(&pkt2);
				}
				audiobuflen -= 1024;
				memmove(audiobuf, audiobuf + 1024, audiobuflen * 2);
			}
		}
	}
	return ret;
}

// Remuxing thread
void *rxthread(void *dummy)
{
	AVPacket pkt;
	int ret = 0;
	char s[300];			

	start_dts = -1;
	// Calculate the fragment size in time base units
	audiobuflen = 0;
	if(fragmentsize_seconds == -1)	// Special case, infinite fragment size (streaming)
		fragmentsize = -1;
	else fragmentsize = fragmentsize_seconds * pFormatCtx->streams[stream_idx]->time_base.den /
		pFormatCtx->streams[stream_idx]->time_base.num;

	// While it's allowed to run
    while (rx_active)
	{
		// Read frame
        ret = av_read_frame(pFormatCtx, &pkt);
        if (ret < 0)
            break;

		log_packet(pFormatCtx, &pkt, "in");
		// If video, decode it
		if(pkt.stream_index == stream_idx) {
			/* decode video frame */
			pthread_mutex_lock(&readmutex);
			avcodec_decode_video2(pCodecCtx, pFrame_yuv, &frame_decoded, &pkt);
			pthread_mutex_unlock(&readmutex);
		}
		if(fragmentsize == 0)
		{
			// We are only receiving and not saving, save the received packets in a FIFO buffer
			struct AVPacket pkt2;
			
			// We have to copy the contents to a new packet, because
			// libav only has a limited amount of packets and after a while
			// they start to cycle and we overrun
			av_new_packet(&pkt2, pkt.size);
			pkt2.stream_index = pkt.stream_index;
			pkt2.dts = pkt.dts;
			pkt2.pts = pkt.pts;
			pkt2.flags = pkt.flags;
			memcpy(pkt2.data, pkt.data, pkt.size);
			rxfifo[rxfifo_tail] = pkt2;
			rxfifo_tail = (rxfifo_tail+1) % RXFIFOQUEUESIZE;
			if(rxfifo_tail == rxfifo_head)
			{
				av_free_packet(&rxfifo[rxfifo_tail]);
				rxfifo_head = (rxfifo_head+1) % RXFIFOQUEUESIZE;
			}
			if((savenow_seconds_before || savenow_seconds_after) && pkt.stream_index == stream_idx)
			{
				int i = rxfifo_tail;
				uint64_t last_dts;
				
				// Go backward for savenow_seconds_before seconds
				uint64_t sdts = pkt.dts - savenow_seconds_before * pFormatCtx->streams[stream_idx]->time_base.den /
					pFormatCtx->streams[stream_idx]->time_base.num;
				if((int64_t)sdts < 0)
					sdts = 0;
				i = rxfifo_tail;
				start_dts = pkt.dts;
				while(i != rxfifo_head)
				{
					i = (i + RXFIFOQUEUESIZE-1) % RXFIFOQUEUESIZE;
					log_packet(pFormatCtx, &rxfifo[i], "going_back");
					if(rxfifo[i].stream_index == stream_idx && rxfifo[i].dts <= sdts)
						break;
				}
				// Go backward until a keyframe is found
				if(!(rxfifo[i].flags & AV_PKT_FLAG_KEY))
				{
					while(i != rxfifo_head)
					{
						i = (i + RXFIFOQUEUESIZE-1) % RXFIFOQUEUESIZE;
						log_packet(pFormatCtx, &rxfifo[i], "going_back_key");
						if(rxfifo[i].stream_index == stream_idx && rxfifo[i].flags & AV_PKT_FLAG_KEY)
							break;
					}
					// If there is no keyframe, go forward and find first keyframe
					while(i != rxfifo_tail && rxfifo[i].stream_index == stream_idx &&
						!(rxfifo[i].flags & AV_PKT_FLAG_KEY))
					{
						log_packet(pFormatCtx, &rxfifo[i], "going_forward");
						i = (i + 1) % RXFIFOQUEUESIZE;
					}
				}
				
				// Open the new file
				ofmt_ctx = openoutput(0, destformat, savenow_path);
				if(!ofmt_ctx)
					return 0;
				if(i != rxfifo_tail)
					start_dts = rxfifo[i].dts;
				last_dts = start_dts;
					
				if(loglevel >= 4)
					fprintf(stderr, "Went back %d seconds, saving %d frames\n", savenow_seconds_before,
						(rxfifo_tail - i + RXFIFOQUEUESIZE) % RXFIFOQUEUESIZE);
				// Write out the buffer
				fragmentsize = 0;
				while(i != rxfifo_tail)
				{
					last_dts = rxfifo[i].dts;
					write_packet(&rxfifo[i], pFormatCtx->streams[rxfifo[i].stream_index]->time_base);
					i = (i + 1) % RXFIFOQUEUESIZE;
				}
				
				// Clear the FIFO
				while(rxfifo_head != rxfifo_tail)
				{
					av_free_packet(&rxfifo[rxfifo_head]);
					rxfifo_head = (rxfifo_head+1) % RXFIFOQUEUESIZE;
				}
				rxfifo_head = rxfifo_tail = 0;

				// Work done, clear the request
				fragmentsize = savenow_seconds_after * pFormatCtx->streams[stream_idx]->time_base.den /
					pFormatCtx->streams[stream_idx]->time_base.num + (last_dts - start_dts);
				if(loglevel >= 4)
					fprintf(stderr, "Savenow: start_dts = %ld, last_dts = %ld, fragmentsize = %ld\n",
						(long)start_dts, (long)last_dts, (long)fragmentsize);
				savenow_seconds_before = savenow_seconds_after = 0;
			}
			av_free_packet(&pkt);
		} else {
			if(savenow_seconds_after && pkt.stream_index == stream_idx)
			{
				fragmentsize = savenow_seconds_after * pFormatCtx->streams[stream_idx]->time_base.den /
					pFormatCtx->streams[stream_idx]->time_base.num + (pkt.dts - start_dts);
				if(loglevel >= 4)
					fprintf(stderr, "Updating savenow: start_dts = %ld, last_dts = %ld, fragmentsize = %ld\n",
						(long)start_dts, (long)pkt.dts, (long)fragmentsize);
				savenow_seconds_before = savenow_seconds_after = 0;
			}
			write_packet(&pkt, pFormatCtx->streams[pkt.stream_index]->time_base);
			av_free_packet(&pkt);
		}
    }

	if(ofmt_ctx)
	{
		// Write the trailer of the file
		av_write_trailer(ofmt_ctx);
		
		/* close output */
		if (ofmt_ctx && !(ofmt_ctx->flags & AVFMT_NOFILE))
		{
			avio_close(ofmt_ctx->pb);
			if(*post_url)
				rx_postfile(ofmt_ctx->filename);
		}
		avformat_free_context(ofmt_ctx);
		ofmt_ctx = 0;

		if (ret < 0 && ret != AVERROR_EOF) {
			av_strerror(ret, s, sizeof(s));
			fprintf(stderr, "Error %d occurred: %s\n", ret, s);
			return 0;
		}
	}
	
	// Clear the FIFO
	while(rxfifo_head != rxfifo_tail)
	{
		av_free_packet(&rxfifo[rxfifo_head]);
		rxfifo_head = (rxfifo_head+1) % RXFIFOQUEUESIZE;
	}
	rxfifo_head = rxfifo_tail = 0;
    return 0;
}

#ifdef DOVIDEOCAP
// Remuxing thread, vcap case
void *rxthread_vcap(void *dummy)
{
	AVRational time_base;
	int ret = 0;
	char s[300];	
	long nframes = 0;

	start_dts = -1;
	time_base.num = 1;
	time_base.den = vcap_fps;
	// Calculate the fragment size in frames
	if(fragmentsize_seconds == -1)	// Special case, infinite fragment size (streaming)
		fragmentsize = -1;
	else fragmentsize = fragmentsize_seconds * vcap_fps;

	// While it's allowed to run
    while (rx_active)
	{
		// Read frame
		char *frame;
		struct timeval tv;
		int keyframe, rc;
		char *outframe;
		unsigned outframelen;
		struct AVPacket pkt;

		// Get the frame from the V4L2 device using our videocap library
		rc = videocap_getframe(vcap, &frame, &tv);
		if(rc < 0)
		{
			fprintf(stderr, "videocap_getframe returned error %d\n", rc);
			break;
		}
		// Save frame for the getframe function
		pthread_mutex_lock(&readmutex);
		memcpy(vcap_frame, frame, vcap_w * vcap_h * 2);
		frame_decoded = 1;
		pthread_mutex_unlock(&readmutex);
		
		// Encode the frame to H.264
		rc = videocodec_process(vcodec, frame, vcap_w * vcap_h * 2, &outframe, &outframelen, &keyframe);
		// keyframe returned by the encoder is totally wrong,
		// so we force a GOP size of 12 and we know that every 12th frame is a keyframe
		if(rc < 0)
		{
			fprintf(stderr, "videocodec_process returned error %d\n", rc);
			break;
		}
		if(!outframelen)
			continue;
		keyframe = vcap_nframes % vcodec_gopsize == 0;
		vcap_nframes++;

		// Put it in a standard libav packet
		av_new_packet(&pkt, outframelen);
		pkt.stream_index = 0;
		pkt.dts = pkt.pts = nframes++;
		pkt.flags = keyframe ? AV_PKT_FLAG_KEY : 0;
		// We have to copy data, because outframe is a pointer to the driver memory,
		// which is no longer valid after the next videocodec_process
		memcpy(pkt.data, outframe, outframelen);

		log_packet(0, &pkt, "in");

		if(fragmentsize == 0)
		{
			// We are only receiving and not saving, save the received packets in a FIFO buffer
			rxfifo[rxfifo_tail] = pkt;
			rxfifo_tail = (rxfifo_tail+1) % RXFIFOQUEUESIZE;
			if(rxfifo_tail == rxfifo_head)
			{
				av_free_packet(&rxfifo[rxfifo_tail]);
				rxfifo_head = (rxfifo_head+1) % RXFIFOQUEUESIZE;
			}
			if(savenow_seconds_before || savenow_seconds_after)
			{
				int i = rxfifo_tail;
				uint64_t last_dts;
				
				// Go backward for savenow_seconds_before seconds
				uint64_t sdts = pkt.dts - savenow_seconds_before * vcap_fps;
				if((int64_t)sdts < 0)
					sdts = 0;
				i = rxfifo_tail;
				start_dts = pkt.dts;
				while(i != rxfifo_head)
				{
					i = (i + RXFIFOQUEUESIZE-1) % RXFIFOQUEUESIZE;
					log_packet(0, &rxfifo[i], "going_back");
					if(rxfifo[i].stream_index == stream_idx && rxfifo[i].dts <= sdts)
						break;
				}
				// Go backward until a keyframe is found
				if(!(rxfifo[i].flags & AV_PKT_FLAG_KEY))
				{
					while(i != rxfifo_head)
					{
						i = (i + RXFIFOQUEUESIZE-1) % RXFIFOQUEUESIZE;
						log_packet(0, &rxfifo[i], "going_back_key");
						if(rxfifo[i].stream_index == stream_idx && rxfifo[i].flags & AV_PKT_FLAG_KEY)
							break;
					}
					// If there is no keyframe, go forward and find first keyframe
					while(i != rxfifo_tail && rxfifo[i].stream_index == stream_idx &&
						!(rxfifo[i].flags & AV_PKT_FLAG_KEY))
					{
						log_packet(0, &rxfifo[i], "going_forward");
						i = (i + 1) % RXFIFOQUEUESIZE;
					}
				}
				
				// Open the new file
				ofmt_ctx = openoutput(0, destformat, savenow_path);
				if(!ofmt_ctx)
					return 0;
				if(i != rxfifo_tail)
					start_dts = rxfifo[i].dts;
				last_dts = start_dts;
				if(loglevel >= 5)
					fprintf(stderr, "openoutput %s last_dts=%ld\n", savenow_path, (long)start_dts);
					
				if(loglevel >= 4)
					fprintf(stderr, "Went back %d seconds, saving %d frames\n", savenow_seconds_before,
						(rxfifo_tail - i + RXFIFOQUEUESIZE) % RXFIFOQUEUESIZE);

				// Write out the buffer
				fragmentsize = 0;
				while(i != rxfifo_tail)
				{
					last_dts = rxfifo[i].dts;
					write_packet(&rxfifo[i], time_base);
					i = (i + 1) % RXFIFOQUEUESIZE;
				}
				
				// Clear the FIFO
				while(rxfifo_head != rxfifo_tail)
				{
					av_free_packet(&rxfifo[rxfifo_head]);
					rxfifo_head = (rxfifo_head+1) % RXFIFOQUEUESIZE;
				}
				rxfifo_head = rxfifo_tail = 0;

				// Work done, clear the request
				fragmentsize = savenow_seconds_after * vcap_fps + (last_dts - start_dts);
				if(loglevel >= 4)
					fprintf(stderr, "Savenow: start_dts = %ld, last_dts = %ld, fragmentsize = %ld\n",
						(long)start_dts, (long)last_dts, (long)fragmentsize);
				savenow_seconds_before = savenow_seconds_after = 0;
			}
		} else {
			if(savenow_seconds_after)
			{
				fragmentsize = savenow_seconds_after * vcap_fps + (pkt.dts - start_dts);
				if(loglevel >= 4)
					fprintf(stderr, "Updating savenow: start_dts = %ld, last_dts = %ld, fragmentsize = %ld\n",
						(long)start_dts, (long)pkt.dts, (long)fragmentsize);
				savenow_seconds_before = savenow_seconds_after = 0;
			}
			write_packet(&pkt, time_base);
			av_free_packet(&pkt);
		}
    }

	if(ofmt_ctx)
	{
		// Write the trailer of the file
		av_write_trailer(ofmt_ctx);
		
		/* close output */
		if (ofmt_ctx && !(ofmt_ctx->flags & AVFMT_NOFILE))
		{
			avio_close(ofmt_ctx->pb);
			if(*post_url)
				rx_postfile(ofmt_ctx->filename);
		}
		avformat_free_context(ofmt_ctx);
		ofmt_ctx = 0;

		if (ret < 0 && ret != AVERROR_EOF) {
			av_strerror(ret, s, sizeof(s));
			fprintf(stderr, "Error %d occurred: %s\n", ret, s);
			return 0;
		}
	}
	
	// Clear the FIFO
	while(rxfifo_head != rxfifo_tail)
	{
		av_free_packet(&rxfifo[rxfifo_head]);
		rxfifo_head = (rxfifo_head+1) % RXFIFOQUEUESIZE;
	}
	rxfifo_head = rxfifo_tail = 0;
    return 0;
}
#endif

// Start a thread that reads frames from the input AVFormatContext and remuxes
// them to the specified file

static int startremux(lua_State *L)
{
	if(rx_tid)
	{
		luaL_error(L, "Another startremux already in progress");
	}
#ifdef DOVIDEOCAP
	if(!pFormatCtx && !(vcap && vcodec))
	{
		if(vcap && !vcodec)
			luaL_error(L, "No codec device was given when capture was called");
		else luaL_error(L, "Call init or capture first");
#else
	if(!pFormatCtx)
	{
		luaL_error(L, "Call init first");
#endif
	}
	strcpy(destfile, lua_tostring(L, 1));
	strcpy(destformat, lua_tostring(L, 2));
	fragmentsize_seconds = lua_tointeger(L, 3);
	if(fragmentsize_seconds != -1)
	{
		char *p;
		
		p = strchr(destfile, '|');
		if(p)
		{
			// We post the file after upload
			*p++ = 0;
			p = strtok(p, "|");
			while(p)
			{
				if(!memcmp(p, "url=", 4))
					strcpy(post_url, p+4);
				else if(!memcmp(p, "username=", 9))
					strcpy(post_username, p+9);
				else if(!memcmp(p, "password=", 9))
					strcpy(post_password, p+9);
				else if(!memcmp(p, "device=", 7))
					strcpy(post_device, p+7);
				p = strtok(0, "|");
			}
		}
		destext = strrchr(destfile, '.');
		if(!destext)
			destext = "";
		else *destext++ = 0;
	} else destext = "";
	// Generated files will be in the form destfile_timestamp.extension
	// Create the first fragment and start the decoding thread
	if(fragmentsize_seconds)
	{
		ofmt_ctx = openoutput(L, destformat, 0);
		if(!ofmt_ctx)
		{
			lua_pushboolean(L, 0);
			return 1;
		}
	}
	rx_active = 1;
	savenow_seconds_before = savenow_seconds_after = 0;
#ifdef DOVIDEOCAP
	if(vcap)
	{
		pthread_create(&rx_tid, 0, rxthread_vcap, 0);
		lua_pushboolean(L, 1);
		return 1;
	}
#endif
	pthread_create(&rx_tid, 0, rxthread, 0);
	lua_pushboolean(L, 1);
	return 1;
}

// Stop the remuxing thread
static int stopremux(lua_State *L)
{
	void *retval;
	
	if(!rx_tid)
	{
		luaL_error(L, "Call startremux first");
	}
	// Tell the thread to stop and wait for it
	rx_active = 0;
	pthread_join(rx_tid, &retval);
	rx_tid = 0;
	lua_pushboolean(L, 1);
	return 1;
}

// Stop the remuxing thread
static int savenow(lua_State *L)
{
	const char *p;
	
	if(!rx_tid)
	{
		luaL_error(L, "Call startremux first");
	}
	savenow_seconds_before = lua_tointeger(L, 1);
	savenow_seconds_after = lua_tointeger(L, 2);
	p = lua_tostring(L, 3);
	if(!p)
		luaL_error(L, "Missing save path");
	strcpy(savenow_path, p);
	if(loglevel >= 5)
		fprintf(stderr, "savenow(%d,%d,%s)\n", savenow_seconds_before, savenow_seconds_after, savenow_path);
	lua_pushboolean(L, 1);
	return 1;
}

// Initialize OpenSSL
static int lua_https_init(lua_State *L)
{
	const char *certfile = lua_tostring(L, 1);
	if(!certfile)
		luaL_error(L, "Missing certificate file");
		
	lua_pushinteger(L, https_init(certfile));
	return 1;
}

// Upload a file with HTTP POST
static int lua_postfile(lua_State *L)
{
	const char *url, *path, *destfilename, *username, *password, *device;
	char retbuf[3000];
	int rc;
	
	url = lua_tostring(L, 1);
	if(!url)
		luaL_error(L, "Missing url");
	path = lua_tostring(L, 2);
	if(!path)
		luaL_error(L, "Missing path");
	destfilename = lua_tostring(L, 3);
	if(!destfilename)
		luaL_error(L, "Missing destfilename");
	username = lua_tostring(L, 4);
	if(!username)
		luaL_error(L, "Missing username");
	password = lua_tostring(L, 5);
	device = lua_tostring(L, 6);
	if(loglevel >= 3)
		fprintf(stderr, "Uploading [%s] to [%s] as [%s] with username=[%s] password=[%s] device=[%s]\n",
			path, url, destfilename, username, password, device);
	rc = postfile(url, path, destfilename, username, password, device, retbuf, sizeof(retbuf));
	if(loglevel >= 3)
		fprintf(stderr, "postfile %s rc=%d (%s)\n%s\n", path, rc, http_error(rc), retbuf);
	lua_pushinteger(L, rc);
	lua_pushstring(L, http_error(rc));
	lua_pushstring(L, retbuf);
	return 3;
}

// Set the logging level
static int lua_loglevel(lua_State *L)
{
	loglevel = lua_tointeger(L, 1);
	return 0;
}

#ifdef DOVIDEOCAP
// Open the capture device and start it with the given parameters
static int videocap_init(lua_State *L)
{
	const char *device = lua_tostring(L, 1);
	int w = lua_tointeger(L, 2);
	int h = lua_tointeger(L, 3);
	int nbuffers = lua_tointeger(L, 5);
	const char *codec = lua_tostring(L, 6);
	int q = lua_tointeger(L, 7);
	int rc;
	int dummy_keyframe;
	char *extradata;
	
	if(vcap)
		videocap_close(vcap);
	vcap_nframes = 0;
	vcap = videocap_open(device);
	vcap_fps = lua_tointeger(L, 4);
	if(!q)
		q = 25;
	if(!vcap)
	{
		luaL_error(L, "Error opening device %s", device);
	}
	if(loglevel >= 3)
		fprintf(stderr, "Starting camera capture at %dx%d, fps=%d, nbuffers=%d\n", w, h, vcap_fps, nbuffers ? nbuffers : 1);
	rc = videocap_startcapture(vcap, w, h, V4L2_PIX_FMT_YUYV, vcap_fps, nbuffers ? nbuffers : 1);
	if(rc < 0)
	{
		videocap_close(vcap);
		vcap = 0;
		luaL_error(L, "Error %d starting capture", rc);
	}
	if(codec)
	{
		vcodec = videocodec_open(codec);
		if(!vcodec)
		{
			videocap_close(vcap);
			vcap = 0;
			luaL_error(L, "Error opening codec device %s", codec);
		}
		rc = videocodec_setcodec(vcodec, V4L2_PIX_FMT_H264);
		// Quantizer, 1-51, lower value means better quality
		// For inter frames, we decrease the quality slightly
		rc = videocodec_setcodecparam(vcodec, V4L2_CID_MPEG_VIDEO_H264_I_FRAME_QP, q);
		rc |= videocodec_setcodecparam(vcodec, V4L2_CID_MPEG_VIDEO_H264_P_FRAME_QP, q+5);
		rc |= videocodec_setcodecparam(vcodec, V4L2_CID_MPEG_VIDEO_H264_B_FRAME_QP, q+5);
		rc |= videocodec_setcodecparam(vcodec, V4L2_CID_MPEG_VIDEO_GOP_SIZE, vcodec_gopsize);
		rc |= videocodec_setcodecparam(vcodec, V4L2_CID_MPEG_VIDEO_H264_LEVEL, V4L2_MPEG_VIDEO_H264_LEVEL_4_0);
		rc |= videocodec_setformat(vcodec, w, h, V4L2_PIX_FMT_YUYV, vcap_fps);
		rc |= videocodec_process(vcodec, (const char *)-1, 0,
			&extradata, (unsigned *)&vcodec_extradata_size, &dummy_keyframe);
		if(rc)
		{
			videocap_close(vcap);
			vcap = 0;
			videocodec_close(vcodec);
			vcodec = 0;
			luaL_error(L, "Error setting encoding parameters");
		}
		vcodec_extradata = malloc(vcodec_extradata_size);
		memcpy(vcodec_extradata, extradata, vcodec_extradata_size);
		vcap_frame = malloc(w * h * 2);
	}
	vcap_w = w;
	vcap_h = h;
	stream_idx = 0; // Required by write_packet
	lua_pushboolean(L, 1);
	return 1;	
}
#endif

/* Availabe functions:

init(file to open, optional path), returns
    status (1=ok, 0=failed)
	width
	height
	number of present frames
	frame rate
	
	Opens a file/stream with libavformat

capture(device_path, width, height[, fps[, nbuffers[, encoder_path, encoder_quality]]]), returns
    status (1=ok, 0=failed)
	
	Opens a video capture device with the videocap library
	This function is only available on Linux
	device_path is in the form /dev/videoN
	fps can be 0 (default)
	default number of buffers is 1
	encoder path is the path of the encoder device
	encoder_quality is the quality of the generated stream (suggested:20-30)
	These two optional parameters are necessary if startremux will be used
	
frame_rgb(type_name, tensor), returns
	status (1=ok, 0=failed)
	
	Gets the next frame in RGB format from the file/stream/device
	type_name can only be torch.ByteTensor
	tensor has to have dimension 3 and the first size has to be 3

frame_yuv(type_name, tensor), returns
	status (1=ok, 0=failed)
	
	Gets the next frame in YUV format from the file/stream
	type_name can only be torch.ByteTensor
	tensor has to have dimension 3 and the first size has to be 3

exit()
	Stops and closes the decoder/video capture device/receiving thread

startremux(fragment_base_path, format, fragment_size), returns
	status (1=ok, 0=failed)
	
	Starts to receive from the stream opened with init and starts to
	write file fragments of size fragment_size seconds (0=don't save
	anything, only save after a savenow command, -1=don't fragment,
	generate a continuous stream, use this when streaming)
	fragment_base_path in the form A.B is changed to A_timestamp.B
	format is the file format (optional), if it cannot be deduced
	from the file extension
	
savenow(seconds before, seconds after, destfilename), returns
	status (1=ok, 0=failed)
	
	Saves the buffered frames from at least now - <seconds before> to
	now + <seconds after>. Receiving should have been started with
	startremux before giving this command. <seconds before> is an "at least"
	value, because saving always starts with a keyframe; of course, it's
	less than this if there is not enough buffered data; destfilename
	is the name of the file that will be saved, both locally and eventually
	on the remote side

stopremux(), returns
	status (1=ok, 0=failed)
	
postfile(url, path, destfilename, username, password, device), returns
	Error code (if positive, it's a HTTP result code)
	Error code in string format
	Contents returned by the server or everything returned by the server if error code != 0
	
	Posts the file path to the url, giving destfilename as the filename parameter,
	passes also the username and password
	
https_init(certificate file), returns
	error code (0=ok, -7=cannot load certificate, -6 SSL/TLS support not compiled in)
	
	Initializes the SSL/TLS library. Requires a certificate for authenticating the server

loglevel(level), no return value
	
	Sets the logging level (0=no logging)
*/


static const struct luaL_reg video_decoder[] = {
	{"init", video_decoder_init},
#ifdef DOVIDEOCAP	
	{"capture", videocap_init},
#endif
	{"frame_rgb", video_decoder_rgb},
	{"frame_yuv", video_decoder_yuv},
	{"exit", video_decoder_exit},
	{"startremux", startremux},
	{"stopremux", stopremux},
	{"savenow", savenow},
	{"postfile", lua_postfile},
	{"https_init", lua_https_init},
	{"loglevel", lua_loglevel},
	{NULL, NULL}
};

// Initialize the library
int luaopen_libvideo_decoder(lua_State * L)
{
	luaL_register(L, "libvideo_decoder", video_decoder);
	/* pre-calculate lookup table */
	video_decoder_yuv420p_rgbp_LUT();
	/* register libav */
	av_register_all();
	avformat_network_init();
	return 1;
}
