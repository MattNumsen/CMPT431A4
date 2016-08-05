#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "hist-equ.h"

__global__ void histAdd(int* hist_device, unsigned char * img_in)
{
    //kernel can run on gpu
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&hist_device[img_in[id]],1);
}

__global__ void parallel_lut(int *lut_device, int *cdf, int *min, int *d){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    lut_device[id] = (int)(((float)cdf[id] - *min)*255/(*d) + 0.5);

    if(lut_device[id] < 0){
        lut_device[id] = 0;
    }
    
}


__global__ void parallel_img_gpu(unsigned char * img_out, unsigned char * img_in, int * lut, int * img_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lut[img_in[idx]] > 255) {
        img_out[idx] = 255;
    }
    else{
        img_out[idx] = (unsigned char)lut[img_in[idx]];
    }
    
}

void gpu_histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    size_t hist_bytes = nbr_bin * sizeof(int);
    size_t img_bytes = img_size * sizeof(unsigned char);

    int* device_hist;
    cudaMalloc(&device_hist, hist_bytes);
    unsigned char* device_image;
    cudaMalloc(&device_image, img_bytes);
    
    for ( int i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    cudaMemcpy(device_hist,hist_out,hist_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_image,img_in,img_bytes, cudaMemcpyHostToDevice);
    //http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#axzz3t1U3YgZc
    int threadsPerBlock = 256;
    int blocksPerGrid = (img_size + threadsPerBlock - 1) / threadsPerBlock;

    histAdd<<<blocksPerGrid,threadsPerBlock>>>(device_hist,device_image);
    cudaMemcpy(hist_out, device_hist, hist_bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(device_hist);
    cudaFree(device_image);

}

void gpu_histogram_equalization(unsigned char * img_out, unsigned char * img_in, int * hist_in, int img_size, int nbr_bin){

    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    size_t lut_bytes = nbr_bin * sizeof(int);
    int i, min, d;
    int cdf_array[nbr_bin];
    int device_cdf_array_bytes = sizeof(cdf_array);

    min = 0;
    i = 0;

    while(min == 0){
        min = hist_in[i++];
    }

    d = img_size - min;

    int *device_lut;
    int *device_min;
    int *device_d;

    int *device_cdf_array;

    cdf_array[0] = hist_in[0];

    for(i = 1; i < nbr_bin; i ++){
        cdf_array[i] = cdf_array[i-1]+hist_in[i];
    }

    cudaMalloc(&device_cdf_array, sizeof(cdf_array));
    cudaMalloc(&device_lut,lut_bytes);
    cudaMalloc(&device_min,sizeof(int));
    cudaMalloc(&device_d,sizeof(int));

    cudaMemcpy(device_cdf_array,cdf_array,device_cdf_array_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_lut,lut,sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);
    cudaMemcpy(device_min,&min,sizeof(int)*1, cudaMemcpyHostToDevice);
    cudaMemcpy(device_d,&d,sizeof(int), cudaMemcpyHostToDevice);


    int threadsPerBlock = 256;
    int blocksPerGrid = (img_size + threadsPerBlock - 1) / threadsPerBlock;
    //int *lut_device, int *cdf, int min, int d
    parallel_lut<<<1,threadsPerBlock>>>(device_lut,device_cdf_array, device_min, device_d);
    cudaMemcpy(lut, device_lut, sizeof(int) * nbr_bin, cudaMemcpyDeviceToHost);


    //cudaMemcpy(device_cdf_array,device_cdf_array,device_cdf_array_bytes, cudaMemcpyHostToDevice);
    //cudaMemcpy(device_lut,lut,sizeof(int)*nbr_bin, cudaMemcpyHostToDevice);
    //cudaMemcpy(device_min,&min,sizeof(int)*1, cudaMemcpyHostToDevice);
    //cudaMemcpy(device_d,&d,sizeof(int), cudaMemcpyHostToDevice);



    unsigned char *device_img_out = 0;
    cudaMalloc(&device_img_out, sizeof(unsigned char) * img_size);
    cudaMemcpy(device_img_out, img_out, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);


    int *device_img_size;
    cudaMalloc(&device_img_size,sizeof(int));
    cudaMemcpy(device_img_size,&img_size,sizeof(int), cudaMemcpyHostToDevice);

    unsigned char* device_image;
    cudaMalloc(&device_image,  sizeof(unsigned char) * img_size);
    cudaMemcpy(device_image, img_in, sizeof(unsigned char) * img_size, cudaMemcpyHostToDevice);
    parallel_img_gpu<<<blocksPerGrid,threadsPerBlock>>>(device_img_out,device_image, device_lut, device_img_size);
    cudaMemcpy(img_out, device_img_out, sizeof(unsigned char) * img_size, cudaMemcpyDeviceToHost);
}
