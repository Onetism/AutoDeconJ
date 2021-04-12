extern "C"
__global__ void psfcomplex(float *d_Src,float *d_Dst,double k,double d,
                                int exsize1,int exsize2)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;

    if((x < exsize1)&&(y < exsize2))
    {
        double src_temp_real =  d_Src[2*(x*exsize2+y)];
        double src_temp_img = d_Src[2*(x*exsize2+y)+1];

        d_Dst[2*(x*exsize2+y)] = (float)((src_temp_real*cos(k*d)-src_temp_img*sin(k*d))/(exsize1*exsize2));
        d_Dst[2*(x*exsize2+y)+1] = (float)((src_temp_real*sin(k*d)+src_temp_img*cos(k*d))/(exsize1*exsize2));
    }
}