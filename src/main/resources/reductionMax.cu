extern "C"
__global__ void reductionMax(float *d_Src,float *d_Dst,float *temp_result,int *blocknum,int blockpergrid,
                        int sizeH1,int sizeH2,int sizeH3,int sizeH4,int sizeH5)
{
    __shared__ float sdata[1024];
    const int tid = threadIdx.x;
    const long x = blockIdx.x*blockDim.x+threadIdx.x;

    long H1 = sizeH1,H2 = sizeH2,H3 = sizeH3,H4 = sizeH4,H5 = sizeH5;

    if(x >= H1*H2*H3*H4*H5)
        return;

    sdata[tid] = d_Src[x];
    if(sdata[tid] < 1e-10)
    {
        sdata[tid] = 0;
    }
    __syncthreads();
    for(int stride = blockDim.x/2;stride>0;stride/=2)
    {
        if(tid< stride)
        {
            if(sdata[tid] < sdata[tid+stride])
            {
                if(sdata[tid+stride] < 1e-10)
                {
                    sdata[tid+stride] = 0;
                }
                sdata[tid] = sdata[tid+stride];
                
            }
        }
        __syncthreads();
    }
    if(tid == 0)
    {
        temp_result[blockIdx.x] = sdata[0];
        __threadfence();
        atomicAdd(blocknum,1);
    }
    if(blocknum[0] == blockpergrid)
    {
        float max = 0.0;
        for(int i = 0; i< blockpergrid; i++)
        {
            if(temp_result[i] > max)
            {
                max = temp_result[i];
            }
        }
        d_Dst[0] = max;
    }  
}