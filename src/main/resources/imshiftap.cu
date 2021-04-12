extern "C"
__global__ void imshiftap(float *d_Dst,float *d_Src,int xmin,int xmax,int ymin,int ymax,
                            int img_h,int img_w,int shiftx,int shifty, int clear)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;

    double temp_real = 0.0;
    double temp_img = 0.0;
    if((x<img_h) && (y<img_w))
    {
        if(clear)
        {
            d_Dst[x * img_w + y] = 0;
        }
        else
        {
            if (shiftx >= 0 & shifty >= 0) 
            {
                long ii = shiftx+x;
                long jj = shifty+y;
                if((ii>=xmin && ii<xmax) && (jj>=ymin && jj<ymax ))
                {
                    temp_real = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))];
                    temp_img = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))+1];
                    d_Dst[ii * img_w + jj] = (float)((1e-20)*sqrt((temp_real*temp_real-temp_img*temp_img)*(temp_real*temp_real-temp_img*temp_img)
                                                    +(2*temp_real*temp_img)*(2*temp_real*temp_img)));
    
                }
          
            } 
            else if (shiftx >= 0 & shifty < 0) 
            {
                long ii = shiftx+x;
                long jj = y;
                if((ii>=xmin && ii<xmax) && (jj>=ymin && jj<ymax ))
                {
                    temp_real = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))];
                    temp_img = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))+1];
                    d_Dst[ii * img_w + jj] = (float)((1e-20)*sqrt((temp_real*temp_real-temp_img*temp_img)*(temp_real*temp_real-temp_img*temp_img)
                                                    +(2*temp_real*temp_img)*(2*temp_real*temp_img)));
                }
    
            } 
            else if (shiftx < 0 & shifty >= 0) 
            {
                long ii = x;
                long jj = shifty+y;
                if((ii>=xmin && ii<xmax) && (jj>=ymin && jj<ymax ))
                {
                    temp_real = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))];
                    temp_img = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))+1];
                    d_Dst[ii * img_w + jj] = (float)((1e-20)*sqrt((temp_real*temp_real-temp_img*temp_img)*(temp_real*temp_real-temp_img*temp_img)
                                                    +(2*temp_real*temp_img)*(2*temp_real*temp_img)));
    
                }
            } 
            else 
            {
                long ii = x;
                long jj = y;        
                if((ii>=xmin && ii<xmax) && (jj>=ymin && jj<ymax ))
                {
                    temp_real = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))];
                    temp_img = d_Src[2*((ii - shiftx) * img_w + (jj - shifty))+1];
                    d_Dst[ii * img_w + jj] = (float)((1e-20)*sqrt((temp_real*temp_real-temp_img*temp_img)*(temp_real*temp_real-temp_img*temp_img)
                                                     +(2*temp_real*temp_img)*(2*temp_real*temp_img)));
    
                }
           
            }
        }
    }

}
