extern "C"
__global__ void normalize(float *d_Src,float max_value,int flag,
                        int sizeH1,int sizeH2,int sizeH3,int sizeH4,int sizeH5)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;

    long H1 = sizeH1;
    long H2 = sizeH2;
    long H3 = sizeH3;
    long H4 = sizeH4;
    long H5 = sizeH5;

    if(x < H1*H2*H3*H4*H5)
    {
        if(flag == 0){
            d_Src[x] = d_Src[x]/max_value;
        }
        if(d_Src[x] < 1e-7)
            d_Src[x] = 0;
    }
}