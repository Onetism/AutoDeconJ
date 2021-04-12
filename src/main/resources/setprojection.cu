extern "C"
__global__ void setprojection(float *d_Dst,int imcenterinit,
                                    int exsize1,int exsize2,int aa,int bb,
                                    int img_h,int img_w)
{
   const long x = blockIdx.x*blockDim.x+threadIdx.x;
   const long y = blockIdx.y*blockDim.y+threadIdx.y;

    if((x < exsize1)&&(y < exsize2))
    {
        d_Dst[2*(x*exsize2+y)] = 0;
        d_Dst[2*(x*exsize2+y)+1] = 0;
        if( (x == (imcenterinit+aa+1)) && (y ==(imcenterinit+bb+1)) )
        {
            d_Dst[2*(x*exsize2+y)] = 1;
        }
       
        
    }
}
