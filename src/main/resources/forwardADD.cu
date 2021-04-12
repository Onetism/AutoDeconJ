extern "C"
__global__ void forwardADD(float *d_Src,float *d_Dst,float *tempdst,int img_h,int img_w,int flag)
{
    int x= blockIdx.x*blockDim.x+threadIdx.x;
    if(x < img_h*img_w)
    {
        if(flag == 0){
            d_Dst[x] = 0;
        }else{
            float t = d_Src[x];
            float h = t - tempdst[x];
            float m = d_Dst[x]+h;
            tempdst[x] = (m-d_Dst[x])-h;
            d_Dst[x] =  m;  

            // d_Dst[x] += d_Src[x];
        }
            
    }


}
