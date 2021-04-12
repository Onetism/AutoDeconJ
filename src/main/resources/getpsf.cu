extern "C"
__global__ void getpsf(float *d_Src,float *d_Dst,int x1shift,int x2shift,int cp0,
                        int sizeH1,int sizeH2,int sizeH3,int sizeH4,int sizeH5,
                        int src_h,int src_w,int z,int b,int a)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;

    long H1 = sizeH1;
    long H2 = sizeH2;
    long H3 = sizeH3;
    long H4 = sizeH4;
    if((x < sizeH1)&&(y < sizeH2))
    {
        d_Dst[(long)(z*H1*H2*H3*H4+ 
                b*H1*H2*H3
                + a*H1*H2+x*H1
                + y)] = d_Src[(x+cp0-x1shift-1)*src_w+(y+cp0-x2shift-1)];        
    }
}