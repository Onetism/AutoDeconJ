extern "C"
__global__ void getHnew(float *d_Src,float *d_Dst,int aa,int bb,int cc,int mmidx,int mmidy,
                        int exsize1,int exsize2,int size_H1,int size_H2,int size_H3,int size_H4,int size_H5,int isHt)
{
    
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;
    if((x<exsize1)&& (y<exsize2)){
        d_Dst[2*(x*exsize2+y)] = 0;
        d_Dst[2*(x*exsize2+y)+1] = 0;
    }
    if((x<exsize1)&&((mmidx+x)%exsize1<size_H1)
        && (y<exsize2)&&((mmidy+y)%exsize2<size_H2))
    {

        if(isHt){
            d_Dst[2*(x*exsize2+y)] = d_Src[cc*size_H4*size_H3*size_H2*size_H1
                                        +bb*size_H3*size_H2*size_H1
                                        +aa*size_H2*size_H1
                                        +(size_H1-1-((mmidx+x)%exsize1))*size_H2
                                        +(size_H2-((mmidy+y)%exsize2)-1)];
        }
        else
        {
            d_Dst[2*(x*exsize2+y)] = d_Src[cc*size_H4*size_H3*size_H2*size_H1
                                            +bb*size_H3*size_H2*size_H1
                                            +aa*size_H2*size_H1
                                            +((mmidx+x)%exsize1)*size_H2
                                            +((mmidy+y)%exsize2)];           
        }


        
    }

}
