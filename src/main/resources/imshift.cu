extern "C"
__global__ void imshift(float *d_Dst,float *d_Src,int z,
                            int img_h,int img_w,int shiftx,int shifty, int clear)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;


    if((x<img_h) && (y<img_w))
    {
        if(clear)
        {
            d_Dst[2*(x * img_w + y)] = 0;
            d_Dst[2*(x * img_w + y)+1] = 0;
        }
        else
        {
            if (shiftx >= 0 & shifty >= 0) 
            {
                long ii = shiftx+x;
                long jj = shifty+y;
                if((ii<img_h) && (jj<img_w))
                {
                    d_Dst[2*(ii * img_w + jj)] = d_Src[z*img_h*img_w*2+2*((ii - shiftx) * img_w + (jj - shifty))];
                    d_Dst[2*(ii * img_w + jj)+1] = d_Src[z*img_h*img_w*2+2*((ii - shiftx) * img_w + (jj - shifty))+1];
    
                }
          
            } 
            else if (shiftx >= 0 & shifty < 0) 
            {
                long ii = shiftx+x;
                long jj = y;
                if((ii<img_h) && (jj<img_w+shifty))
                {
                    d_Dst[2*(ii * img_w + jj)] = d_Src[z*img_h*img_w*2 + 2*((ii - shiftx) * img_w + (jj - shifty))];
                    d_Dst[2*(ii * img_w + jj)+1] = d_Src[z*img_h*img_w*2 + 2*((ii - shiftx) * img_w + (jj - shifty))+1];
    
                }
    
            } 
            else if (shiftx < 0 & shifty >= 0) 
            {
                long ii = x;
                long jj = shifty+y;
                if((ii<img_h+shiftx)&&(jj<img_w))
                {
                    d_Dst[2*(ii * img_w + jj)] = d_Src[z*img_h*img_w*2 + 2*((ii - shiftx) * img_w + (jj - shifty))];
                    d_Dst[2*(ii * img_w + jj)+1] = d_Src[z*img_h*img_w*2 + 2*((ii - shiftx) * img_w + (jj - shifty))+1];
    
                }
            } 
            else 
            {
                long ii = x;
                long jj = y;        
                if((ii<img_h+shiftx)&&(jj<img_w+shifty))
                {
                    d_Dst[2*(ii * img_w + jj)] = d_Src[z*img_h*img_w*2 + 2*((ii - shiftx) * img_w + (jj - shifty))];
                    d_Dst[2*(ii * img_w + jj)+1] = d_Src[z*img_h*img_w*2 + 2*((ii - shiftx) * img_w + (jj - shifty))+1];
    
                }
           
            }
        }


    }

}
