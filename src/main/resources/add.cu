extern "C"
__global__ void add(float *d_Src,float *d_Dst, float *tempdst, int aa,int bb,int cc,int Nnum,
                    int exsize1,int exsize2,int img_h,int img_w,int flag)
{
    int x= blockIdx.x*blockDim.x+threadIdx.x;
    int y= blockIdx.y*blockDim.y+threadIdx.y;

    if(flag==1)
    {
        long i = aa+x;
        long j = bb+y;
    
        if((aa==0)&&(bb==0)&&(i<img_h)&&(j<img_w))
        {
            d_Dst[cc*img_h*img_w+i*img_w+j] = 0;
        }
        if((i < img_h)&&(j < img_w)
            &&(x%Nnum == 0)&&((y%Nnum == 0)))
        {
            if(d_Src[2*(i*exsize2+j)] < 1e-10){
                d_Src[2*(i*exsize2+j)] = 0;
            }
            // d_Dst[cc*img_h*img_w+i*img_w+j] += d_Src[2*(i*exsize2+j)]/(exsize1*exsize2);

            float t = d_Src[2*(i*exsize2+j)]/(exsize1*exsize2);
            float h = t - tempdst[cc*img_h*img_w+i*img_w+j];
            float m = d_Dst[cc*img_h*img_w+i*img_w+j]+h;
            tempdst[cc*img_h*img_w+i*img_w+j] = (m-d_Dst[cc*img_h*img_w+i*img_w+j])-h;
            d_Dst[cc*img_h*img_w+i*img_w+j] =  m;  
            
        }
    }
    else if(flag == 2)
    {
        if((x<img_h)&&(y<img_w))
        { 
            float t = d_Src[2*(x*exsize2+y)]/(exsize1*exsize2);
            float h = t - tempdst[x*img_w+y];
            float m = d_Dst[x*img_w+y]+h;
            tempdst[x*img_w+y] = (m-d_Dst[x*img_w+y])-h;
            d_Dst[x*img_w+y] =  m; 
            // double t = (double)d_Src[2*(x*exsize2+y)]/((double)exsize1*exsize2);
            // tempdst[x*img_w+y] +=t;
            // d_Dst[x*img_w+y] = tempdst[x*img_w+y];
            // d_Dst[x*img_w+y] = t;
        }
    }
    else if(flag == 3)
    {
        if((x<img_h)&&(y<img_w))
        {   
            if(d_Src[2*(x*exsize2+y)] < 1e-10){
                d_Src[2*(x*exsize2+y)] = 0;
            }
            // double t = (double)d_Src[2*(x*exsize2+y)]/((double)exsize1*exsize2);
            // tempdst[cc*img_h*img_w+x*img_w+y] += t;
            // d_Dst[cc*img_h*img_w+x*img_w+y] = tempdst[cc*img_h*img_w+x*img_w+y];

            // d_Dst[cc*img_h*img_w+x*img_w+y] =  tempdst[cc*img_h*img_w+x*img_w+y];
            float t = d_Src[2*(x*exsize2+y)]/(exsize1*exsize2);
            float h = t - tempdst[cc*img_h*img_w+x*img_w+y];
            float m = d_Dst[cc*img_h*img_w+x*img_w+y]+h;
            tempdst[cc*img_h*img_w+x*img_w+y] = (m-d_Dst[cc*img_h*img_w+x*img_w+y])-h;
            d_Dst[cc*img_h*img_w+x*img_w+y] =  m;     

            // d_Dst[cc*img_h*img_w+x*img_w+y] += d_Src[2*(x*exsize2+y)]/(exsize1*exsize2);
        }      
    }

}
