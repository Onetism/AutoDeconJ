extern "C"
__global__ void multiply(float *d_Src,float *d_Mask,float *d_Dst,
                                    int exsize1,int exsize2)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;

    if((x < exsize1)&&(y < exsize2))
    {
        float src_temp_real =  d_Src[2*(x*exsize2+y)];
        float src_temp_img = d_Src[2*(x*exsize2+y)+1];
    
        float mask_temp_real =  d_Mask[2*(x*exsize2+y)];
        float mask_temp_img = d_Mask[2*(x*exsize2+y)+1]; 

        

        d_Dst[2*(x*exsize2+y)] = src_temp_real*mask_temp_real-src_temp_img*mask_temp_img;
        d_Dst[2*(x*exsize2+y)+1] = src_temp_real*mask_temp_img+src_temp_img*mask_temp_real;
    }
}
