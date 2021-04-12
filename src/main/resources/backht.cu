extern "C"
__global__ void backht(float *d_Dst,float *d_Src,int sizeH1,int sizeH2,
                            int sizeH3,int sizeH4,int bb,int aa,int imcenter,
                            int src_h,int src_w,
                            int shiftx,int shifty,int sizeH5)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;
    const long z = blockIdx.z*blockDim.z+threadIdx.z;
    if((x<sizeH1) && (y<sizeH2) && (z<sizeH5))
    {  
        if (shiftx >= 0 & shifty >= 0) 
        {
            long ii = shiftx+x;
            long jj = shifty+y;
            if((ii<sizeH1) && (jj<sizeH2))
            {
                d_Dst[z*sizeH4*sizeH3*sizeH2*sizeH1
                    +bb*sizeH3*sizeH2*sizeH1
                    +aa*sizeH2*sizeH1
                    +ii*sizeH2+jj] 
                    = d_Src[z*src_h*src_w+(imcenter-(sizeH1-1)/2+ii-shiftx)*src_w 
                        + (imcenter-(sizeH1-1)/2+jj-shifty)];
  
               

            }
        
        } 
        else if (shiftx >= 0 & shifty < 0) 
        {
            long ii = shiftx+x;
            long jj = y;
            if((ii<sizeH1) && (jj<sizeH2+shifty))
            {
                d_Dst[z*sizeH4*sizeH3*sizeH2*sizeH1
                    +bb*sizeH3*sizeH2*sizeH1
                    +aa*sizeH2*sizeH1
                    +ii*sizeH2+jj] 
                    = d_Src[z*src_h*src_w+(imcenter-(sizeH1-1)/2+ii-shiftx)*src_w 
                        + (imcenter-(sizeH1-1)/2+jj-shifty)];
                

            }

        } 
        else if (shiftx < 0 & shifty >= 0) 
        {
            long ii = x;
            long jj = shifty+y;
            if((ii<sizeH1+shiftx)&&(jj<sizeH2))
            {
                d_Dst[z*sizeH4*sizeH3*sizeH2*sizeH1
                    +bb*sizeH3*sizeH2*sizeH1
                    +aa*sizeH2*sizeH1
                    +ii*sizeH2+jj] 
                    = d_Src[z*src_h*src_w+(imcenter-(sizeH1-1)/2+ii-shiftx)*src_w 
                        + (imcenter-(sizeH1-1)/2+jj-shifty)];
              

            }
        } 
        else 
        {
            long ii = x;
            long jj = y;        
            if((ii<sizeH1+shiftx)&&(jj<sizeH2+shifty))
            {
                d_Dst[z*sizeH4*sizeH3*sizeH2*sizeH1
                    +bb*sizeH3*sizeH2*sizeH1
                    +aa*sizeH2*sizeH1
                    +ii*sizeH2+jj] 
                    = d_Src[z*src_h*src_w+(imcenter-(sizeH1-1)/2+ii-shiftx)*src_w 
                        + (imcenter-(sizeH1-1)/2+jj-shifty)];

            }
        }
    }
}
