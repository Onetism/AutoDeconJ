extern "C"
__global__ void getpsf_fromGPU(float *d_Src,float *d_Dst,int z,
                        int sizeH1,int sizeH2,int sizeH3,int sizeH4)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    // const long y = blockIdx.y*blockDim.y+threadIdx.y;

    long H1 = sizeH1;
    long H2 = sizeH2;
    long H3 = sizeH3;
    long H4 = sizeH4;
    if(x < H1*H2*H3*H4)
    {
        d_Dst[x] = d_Src[z*H1*H2*H3*H4+x];   
        // d_Dst[x] = d_Src[x];     
    }
}