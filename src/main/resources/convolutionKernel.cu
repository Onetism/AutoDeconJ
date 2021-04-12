extern "C"
__global__ void convolutionKernel(float *d_Src,float *d_Mask,float *d_Dst,
                                    int imageW,int imageH,int maskW,int maskH)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;

    if(x < imageH && y < imageW)
    {
        d_Dst[x*imageW+y] = 0.0;
        long startx = x+maskH/2;
        long starty = y+maskW/2;        
    
        for(int i=0;i<maskH;i++)
        {
            for(int j=0;j<maskW;j++)
            {
                if(startx-i >=0 && startx-i < imageH &&starty-j >=0 && starty-j < imageW)
                {
                    d_Dst[x*imageW+y] += d_Src[(startx-i)*imageW+starty-j]*d_Mask[i*maskW+j];
                }                   	    
            }
        }	
    }
}
