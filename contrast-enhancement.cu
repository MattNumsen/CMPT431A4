#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"
#include "gpu_histogram.cuh"

__device__ void GPU_Hue_2_RGB(float &output, float v1, float v2, float vH){        //Function Hue_2_RGB
    if (vH < 0){ 
    	vH += 1;
    }
    if (vH > 1){
    	vH -= 1;
    }
    if ((6 * vH) < 1){
    	output = ( v1 + ( v2 - v1 ) * 6 * vH );
    } else if ((2 * vH) < 1) {
    	output = v2;
    } else if ((3 * vH) < 2) {
    	output = (v1 + (v2 - v1) * ((2.0f/3.0f) - vH) * 6);
    } else {
    	output = v1;	
    }
}

__global__ void gpu_calc_rgb2yuv(unsigned char * img_in_r, unsigned char * img_in_g, unsigned char * img_in_b, unsigned char * img_out_y, unsigned char * img_out_u, unsigned char * img_out_v){
	int id = blockIdx.x * blockDim.x + threadIdx.x;  
	img_out_y[id]  = (unsigned char)( 0.299*img_in_r[id] + 0.587*img_in_g[id] +  0.114*img_in_b[id]);
	img_out_u[id] = (unsigned char)(-0.169*img_in_r[id] - 0.331*img_in_g[id] +  0.499*img_in_b[id] + 128);
	img_out_v[id] = (unsigned char)( 0.499*img_in_r[id] - 0.418*img_in_g[id] - 0.0813*img_in_b[id] + 128);	
}

__global__ void gpu_calc_yuv2rgb(unsigned char * img_in_y, unsigned char * img_in_u, unsigned char * img_in_v, unsigned char * img_out_r, unsigned char * img_out_g, unsigned char * img_out_b){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int y =  (int)img_in_y[id];
	int cb = (int)img_in_u[id] - 128;
    int cr = (int)img_in_v[id] - 128;
    int rt  = (int)( y + 1.402*cr);
    int gt  = (int)( y - 0.344*cb - 0.714*cr);
    int bt  = (int)( y + 1.772*cb);
    
    if (rt > 255){
    	rt = 255;
    } else if (rt < 0){
    	rt = 0;
    }
    if (gt > 255){
    	gt = 255;
    } else if (gt < 0){
    	gt = 0;
    }
    if (bt > 255){
    	bt = 255;
    } else if (bt < 0){
    	bt = 0;
    }
    img_out_r[id] = (unsigned char) rt;
    img_out_g[id] = (unsigned char) gt;
    img_out_b[id] = (unsigned char) bt;
}

__global__ void gpu_calc_rgb2hsl(unsigned char * img_in_r, unsigned char * img_in_g, unsigned char * img_in_b, float * img_out_h, float * img_out_s, unsigned char * img_out_l) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	float var_r = ( (float)img_in_r[id]/255 );//Convert RGB to [0,1]
	float var_g = ( (float)img_in_g[id]/255 );
   	float var_b = ( (float)img_in_b[id]/255 );
   	float var_min = (var_r < var_g) ? var_r : var_g;
   	var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
	float var_max = (var_r > var_g) ? var_r : var_g;
	var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
	float del_max = var_max - var_min;               //Delta RGB value
    float H, S, L;
    L = ( var_max + var_min ) / 2;
    
	if ( del_max == 0 ) {//This is a gray, no chroma...
		H = 0;         
		S = 0;    
	} else {                                   //Chromatic data...
		if ( L < 0.5 ){
			S = del_max/(var_max+var_min);
		} else {
			S = del_max/(2-var_max-var_min );
		}
            
		float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
		float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
		float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
		if( var_r == var_max ) {
			H = del_b - del_g;
		} else {       
			if( var_g == var_max ){
				H = (1.0/3.0) + del_r - del_b;
			} else {
                	H = (2.0/3.0) + del_g - del_r;
                }   
            }
        }
       
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;
        img_out_h[id] = H;
        img_out_s[id] = S;
        img_out_l[id] = (unsigned char)(L*255); 
}

__global__ void gpu_calc_hsl2rgb(float * img_in_h, float * img_in_s, unsigned char * img_in_l, unsigned char * img_out_r, unsigned char * img_out_g, unsigned char * img_out_b) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	
	float H = img_in_h[id];
	float S = img_in_s[id];
	float L = (float)img_in_l[id]/255.0f;
    float var_1, var_2;
    
    float rr, gg, bb;
     
    unsigned char r,g,b;
     
    if ( S == 0 ){
        r = L * 255;
        g = L * 255;
        b = L * 255;
    } else {
        if ( L < 0.5 )
            var_2 = L * ( 1 + S );
        else
            var_2 = ( L + S ) - ( S * L );
        var_1 = 2 * L - var_2;
        GPU_Hue_2_RGB(rr, var_1, var_2, H + (1.0f/3.0f) );
        GPU_Hue_2_RGB(gg, var_1, var_2, H);
        GPU_Hue_2_RGB(bb, var_1, var_2, H - (1.0f/3.0f) );
        r = 255 * rr;
        g = 255 * gg;
        b = 255 * bb;
    }
    img_out_r[id] = r;
    img_out_g[id] = g;
    img_out_b[id] = b;
}


PGM_IMG gpu_contrast_enhancement_g(PGM_IMG img_in) //greyscale
{
    PGM_IMG result;
    int hist[256];
    
    result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
    gpu_histogram(hist, img_in.img, img_in.h * img_in.w, 256);
    gpu_histogram_equalization(result.img,img_in.img,hist,result.w*result.h, 256);
    
    return result;
}

PPM_IMG contrast_enhancement_g_yuv(PPM_IMG img_in) //yuv image technique
{
    YUV_IMG yuv_med;
    PPM_IMG result;
   
    unsigned char * y_equ;
    int hist[256];
  
    yuv_med = gpu_rgb2yuv(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));
  
    gpu_histogram(hist, yuv_med.img_y, yuv_med.h * yuv_med.w, 256);
    gpu_histogram_equalization(y_equ,yuv_med.img_y,hist,yuv_med.h * yuv_med.w, 256);
    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
   
    result = gpu_yuv2rgb(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
   
    return result;
}

PPM_IMG contrast_enhancement_g_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
  
    unsigned char * l_equ;
    int hist[256];

    hsl_med = gpu_rgb2hsl(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));

    gpu_histogram(hist, hsl_med.l, hsl_med.height * hsl_med.width, 256);
    gpu_histogram_equalization(l_equ, hsl_med.l,hist,hsl_med.width*hsl_med.height, 256);
   
    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = gpu_hsl2rgb(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    return result;
}

//Convert HSL to RGB, assume H, S in [0.0, 1.0] and L in [0, 255]
//Output R,G,B in [0, 255]
PPM_IMG gpu_hsl2rgb(HSL_IMG img_in)
{
    PPM_IMG img_out;
 
    img_out.w = img_in.width;
    img_out.h = img_in.height;
    img_out.img_r = (unsigned char *)malloc(img_out.w * img_out.h * sizeof(unsigned char));
    img_out.img_g = (unsigned char *)malloc(img_out.w * img_out.h * sizeof(unsigned char));
    img_out.img_b = (unsigned char *)malloc(img_out.w * img_out.h * sizeof(unsigned char));
	
	size_t img_channel_size = sizeof(unsigned char)*img_in.width*img_in.height;
	size_t hs_size = img_in.width * img_in.height * sizeof(float);
	
	unsigned char * device_r; //gpu variables
	unsigned char * device_g;
	unsigned char * device_b;
	float * device_h;
	float * device_s;
	unsigned char * device_l;
	
    cudaMalloc(&device_r, img_channel_size);
    cudaMalloc(&device_g, img_channel_size);
    cudaMalloc(&device_b, img_channel_size);
    
    cudaMalloc(&device_h, hs_size);
    cudaMalloc(&device_s, hs_size);
    cudaMalloc(&device_l, img_channel_size);
    
    cudaMemcpy(device_r, img_out.img_r, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_g, img_out.img_g, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, img_out.img_b, img_channel_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(device_h, img_in.h, hs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_s, img_in.s, hs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_l, img_in.l, img_channel_size, cudaMemcpyHostToDevice);
 
	//start GPU code execution here
	int threadsPerBlock = 256;
    int blocksPerGrid = (img_channel_size + threadsPerBlock - 1) / threadsPerBlock;
    gpu_calc_hsl2rgb<<<blocksPerGrid,threadsPerBlock>>>(device_h, device_s, device_l,  device_r, device_g, device_b);

    cudaMemcpy(img_out.img_r, device_r, img_channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, device_g, img_channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, device_b, img_channel_size, cudaMemcpyDeviceToHost);
    
    return img_out;
}

HSL_IMG gpu_rgb2hsl(PPM_IMG img_in)
{
    HSL_IMG img_out;// = (HSL_IMG *)malloc(sizeof(HSL_IMG));
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));
    
    size_t img_channel_size = sizeof(unsigned char)*img_in.w*img_in.h;
	size_t hs_size = img_in.w * img_in.h * sizeof(float);
	
	unsigned char * device_r; //gpu variables
	unsigned char * device_g;
	unsigned char * device_b;
	float * device_h;
	float * device_s;
	unsigned char * device_l;
	
    cudaMalloc(&device_r, img_channel_size);
    cudaMalloc(&device_g, img_channel_size);
    cudaMalloc(&device_b, img_channel_size);
    
    cudaMalloc(&device_h, hs_size);
    cudaMalloc(&device_s, hs_size);
    cudaMalloc(&device_l, img_channel_size);
    
    cudaMemcpy(device_r, img_in.img_r, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_g, img_in.img_g, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, img_in.img_b, img_channel_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(device_h, img_out.h, hs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_s, img_out.s, hs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_l, img_out.l, img_channel_size, cudaMemcpyHostToDevice);
    
    //start GPU code execution here
	int threadsPerBlock = 256;
    int blocksPerGrid = (img_channel_size + threadsPerBlock - 1) / threadsPerBlock;
    gpu_calc_rgb2hsl<<<blocksPerGrid,threadsPerBlock>>>(device_r, device_g, device_b,  device_h, device_s, device_l);
    
    //finish GPU code execution
    cudaMemcpy(img_out.h, device_h, hs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.s, device_s, hs_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.l, device_l, img_channel_size, cudaMemcpyDeviceToHost);
    
    return img_out;
}

 
 
YUV_IMG gpu_rgb2yuv(PPM_IMG img_in)
{
	YUV_IMG img_out; //local copy
	    
	img_out.w = img_in.w;
	img_out.h = img_in.h;
	img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
	img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
	img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
   
   //This is where we call the cuda stuff
    
	size_t img_channel_size = sizeof(unsigned char)*img_in.w*img_in.h;
	
	unsigned char * device_r; //gpu variables
	unsigned char * device_g;
	unsigned char * device_b;
	unsigned char * device_y;
	unsigned char * device_u;
	unsigned char * device_v;
	
    cudaMalloc(&device_r, img_channel_size);
    cudaMalloc(&device_g, img_channel_size);
    cudaMalloc(&device_b, img_channel_size);
    
    cudaMalloc(&device_y, img_channel_size);
    cudaMalloc(&device_u, img_channel_size);
    cudaMalloc(&device_v, img_channel_size);
    
    cudaMemcpy(device_r, img_in.img_r, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_g, img_in.img_g, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, img_in.img_b, img_channel_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(device_y, img_out.img_y, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_u, img_out.img_u, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, img_out.img_v, img_channel_size, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (img_channel_size + threadsPerBlock - 1) / threadsPerBlock;
    
    gpu_calc_rgb2yuv<<<blocksPerGrid,threadsPerBlock>>>(device_r, device_g, device_b,  device_y, device_u, device_v);
    
    cudaMemcpy(img_out.img_y, device_y, img_channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, device_u, img_channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, device_v, img_channel_size, cudaMemcpyDeviceToHost);
    
    cudaFree(device_r);
    cudaFree(device_g);
    cudaFree(device_b);
    
    cudaFree(device_y);
    cudaFree(device_u);
    cudaFree(device_v);
    
    return img_out;
}
PPM_IMG gpu_yuv2rgb(YUV_IMG img_in)
{
	PPM_IMG img_out;
      
	img_out.w = img_in.w;
	img_out.h = img_in.h;
    
	img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
	img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
	img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

	size_t img_channel_size = sizeof(unsigned char)*img_in.w*img_in.h;
	
	unsigned char * device_r;//GPU variables for OUTPUT
	unsigned char * device_g;
	unsigned char * device_b;
	
	unsigned char * device_y;//GPU variables for INPUT
	unsigned char * device_u;
	unsigned char * device_v;
	
	cudaMalloc(&device_r, img_channel_size);
    cudaMalloc(&device_g, img_channel_size);
    cudaMalloc(&device_b, img_channel_size);
    
    cudaMalloc(&device_y, img_channel_size);
    cudaMalloc(&device_u, img_channel_size);
    cudaMalloc(&device_v, img_channel_size);
    
	cudaMemcpy(device_r, img_out.img_r, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_g, img_out.img_g, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, img_out.img_b, img_channel_size, cudaMemcpyHostToDevice);
    
    cudaMemcpy(device_y, img_in.img_y, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_u, img_in.img_u, img_channel_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_v, img_in.img_v, img_channel_size, cudaMemcpyHostToDevice);
	 //run functions here
	int threadsPerBlock = 256;
    int blocksPerGrid = (img_channel_size + threadsPerBlock - 1) / threadsPerBlock;
    
    gpu_calc_yuv2rgb<<<blocksPerGrid,threadsPerBlock>>>(device_y, device_u, device_v, device_r, device_g, device_b);
     //done running on GPU
    cudaMemcpy(img_out.img_r, device_r, img_channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, device_g, img_channel_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, device_b, img_channel_size, cudaMemcpyDeviceToHost);
    
    cudaFree(device_r);
    cudaFree(device_g);
    cudaFree(device_b);
    
    cudaFree(device_y);
    cudaFree(device_u);
    cudaFree(device_v);
    
     return img_out;
 }

/*
 float Hue_2_RGB( float v1, float v2, float vH )             //Function Hue_2_RGB
 {
     if ( vH < 0 ) vH += 1;
     if ( vH > 1 ) vH -= 1;
     if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
     if ( ( 2 * vH ) < 1 ) return ( v2 );
     if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
     return ( v1 );
 }


*/
