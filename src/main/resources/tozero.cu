extern "C"
__global__ void tozero(float *src,int length)
{
    const int x = blockIdx.x*blockDim.x+threadIdx.x;

    if(x >= length){
        return;
    }
    src[x] = 0;
}