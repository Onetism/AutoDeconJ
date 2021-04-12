extern "C"

__global__ void xguessmax(float *Htf,float* hxguessback,float *xguess,float *temp_result,float* maxout,
                        int *blocknum,int blockpergrid,long length)
{
    __shared__ float sdata[1024];
    // static int max = 0;
    const int tid = threadIdx.x;
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    // unsigned int block_max = blockpergrid;
    if(x >= length){
        return;
    }
        
    sdata[tid] = xguess[x]*(Htf[x]/hxguessback[x]);
    // if(x==142123519){
    //     printf(" xguess[x] = %.15f,Htf[x] = %.15f, hxguessback[x]= %.15f, sdata[tid] = %.15f\n",xguess[x],Htf[x],hxguessback[x],sdata[tid]);
    // }
    if(sdata[tid]<0 || isnan(sdata[tid]) || isinf(sdata[tid]))
    {
        sdata[tid] = 0;    
    }
    xguess[x] = sdata[tid];
    // if(x==0){
    //     printf("%f\n",sdata[tid]);
    // }
    __syncthreads();

    for(int stride = blockDim.x/2;stride>0;stride/=2)
    {
        if(tid< stride)
        {
            if(sdata[tid] < sdata[tid+stride])
            {
                sdata[tid] = sdata[tid+stride];
                
            }
        }
        __syncthreads();
    }

    if(tid == 0)
    {
        temp_result[blockIdx.x] = sdata[0];
        // temp_result[blockIdx.x] = 100;
        __threadfence();
        atomicAdd(blocknum,1);

    }
    if(blocknum[0] == blockpergrid){
        float max = 0.0;
        for(int i = 0; i< blockpergrid;i++)
        {
            if(temp_result[i] > max)
            {
                max = temp_result[i];
            }
        }
        maxout[0] = max;
    }

}