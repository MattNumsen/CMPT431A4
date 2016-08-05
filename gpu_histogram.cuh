#include "hist-equ.h"
void gpu_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin);
PGM_IMG gpu_contrast_enhancement_g(PGM_IMG img_in);
PPM_IMG contrast_enhancement_g_yuv(PPM_IMG img_in);
PPM_IMG contrast_enhancement_g_hsl(PPM_IMG img_in);
HSL_IMG gpu_rgb2hsl(PPM_IMG img_in);
//float Hue_2_RGB( float v1, float v2, float vH );
PPM_IMG gpu_hsl2rgb(HSL_IMG img_in);
YUV_IMG gpu_rgb2yuv(PPM_IMG img_in);
//unsigned char clip_rgb(int x);
PPM_IMG gpu_yuv2rgb(YUV_IMG img_in);
