extern "C"
__global__ void getprojection(float *d_Src, float *d_Dst,int Nnum,
                                    int exsize1,int exsize2,int aa,int bb,int cc,
                                    int img_h,int img_w,int flag)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;
    if(flag == 0){
        if((x<exsize1)&&(y<exsize2)){
            d_Dst[2*(x*exsize2+y)] = 0;
            d_Dst[2*(x*exsize2+y)+1] = 0;
        }
    }
    else if(flag == 1)
    {
        long i = aa+x;
        long j = bb+y;
        if((i < img_h)&&(j < img_w)
            &&(x%Nnum==0)&&(y%Nnum==0))
            
        {
            d_Dst[2*(i*exsize2+j)] = d_Src[cc*img_h*img_w+i*img_w+j];   
        }
    }
    else if(flag == 2)
    {
        long i = aa+x;
        long j = bb+y;
        if((i < img_h)&&(j < img_w)
            &&(x%Nnum==0)&&(y%Nnum==0))
            
        {
            d_Dst[2*(i*exsize2+j)] = d_Src[i*img_w+j];   
        }      
    }



}
