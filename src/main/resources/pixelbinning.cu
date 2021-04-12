extern "C"
__global__ void pixelbinning(float *d_Src,float *d_Dst,int x1init,int x2init,
                                int src_h,int src_w,int dst_h,int dst_w,int OSR,int flag)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;

    if((x < dst_h)&&(y < dst_w))
    {
        if(flag == 1)
        {
            d_Dst[x*dst_w+y] = 0;
        }
        else
        {
            for(int i = x*OSR; i< (x+1)*OSR ;i++)
            {
                for(int j = y*OSR; j< (y+1)*OSR ;j++)
                {
                    d_Dst[x*dst_w+y] = d_Dst[x*dst_w+y]+d_Src[(i+x1init)*src_w+(j+x2init)];
                }
            } 
        }
              
    }
}