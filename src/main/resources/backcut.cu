extern "C"
__global__ void backcut(float *d_Dst,float *d_Src,int imcenter,int sizeH1,
                            int sizeH2,int src_h,int src_w,int sizeH5)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;
    const long z = blockIdx.z*blockDim.z+threadIdx.z;
    if((x<sizeH1) && (y<sizeH2) && (z<sizeH5))
    {  
        d_Dst[z*sizeH2*sizeH1+x*sizeH2+y] 
        = d_Src[z*src_h*src_w+(imcenter-(sizeH1-1)/2+x)*src_w 
                    + (imcenter-(sizeH1-1)/2+y)];
               
    }
}
