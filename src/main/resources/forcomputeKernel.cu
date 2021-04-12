extern "C"
__global__ void forcomputeKernel(double *x1space,double *x2space,double *pattern_real,double *pattern_img,
                                int centerPT,double alpha,double M,double k,int x1length,int x2length,
                                double p3,int centerarea_il_start,double fobj,double lambda)
{
    const long x = blockIdx.x*blockDim.x+threadIdx.x;
    const long y = blockIdx.y*blockDim.y+threadIdx.y;

    long a = centerarea_il_start+x;
    long b = a+y;
    if(a<=centerPT && b<=centerPT)
    {
        double x1 = x1space[a];
        double x2 = x2space[b];

        double xL2normsq = sqrt((x1)*(x1)+(x2)*(x2))/M;
        double v = k*xL2normsq*sin(alpha);
        double u= 4.0*k*p3*(sin(alpha/2.0)*sin(alpha/2.0));;
        long t = (long)(u*1e12);
        u = t*1e-12;

        double Koi = M/(fobj*lambda*fobj*lambda);
        double ku1 = -1.0*u/(4.0*(sin(alpha/2.0)*sin(alpha/2.0)));
        // if(x==0 && y==0){
        //     printf("u=%.20f   ku1=%.20f\n", u,ku1) ;
        // } 
        double cosk = 0;
        double sink = 0;
        
        double alphaspace = alpha/10240;		
        for (int j = 0; j < 10240; j++){
            double ku2 = -1.0*u*(sin(j*alphaspace/2)*sin(j*alphaspace/2))/(2*(sin(alpha/2)*sin(alpha/2)));

            double tempx = sin(j*alphaspace)*v/sin(alpha);
            if (tempx < 0) tempx *= -1;
                 
            double J0 = j0(tempx);
            double ku3 = Koi*sqrt(cos(j*alphaspace))*(1.0+cos(j*alphaspace))
                        *J0*sin(j*alphaspace)*alphaspace;
            cosk += ku3*cos(ku2+ku1);		//real part in complex
            sink += ku3*sin(ku2+ku1);		//imaginary part in complex
        }
        // if(abs(sink)<1e-10 && cosk < 1e-10)
        // {
        //     cosk = 0;
        //     sink = 0;
        // }
        pattern_real[a*x2length+b] = cosk;
        pattern_real[a*x2length+x2length-1-b] = cosk;
        pattern_real[b*x2length+x2length-1-a] = cosk;
        pattern_real[b*x2length+a] = cosk;

        pattern_real[(x2length-1-b)*x2length+a] = cosk;
        pattern_real[(x2length-1-a)*x2length+x1length-1-b] = cosk;
        pattern_real[(x2length-1-b)*x2length+x1length-1-a] = cosk;
        pattern_real[(x1length-1-a)*x2length+b] = cosk;        

        pattern_img[a*x2length+b] = sink;
        pattern_img[a*x2length+x2length-1-b] = sink;
        pattern_img[b*x2length+x1length-1-a] = sink;
        pattern_img[b*x2length+a] = sink;
        pattern_img[(x2length-1-b)*x2length+a] = sink;
        pattern_img[(x2length-1-a)*x2length+x1length-1-b] = sink;
        pattern_img[(x2length-1-b)*x2length+x1length-1-a] = sink;
        pattern_img[(x1length-1-a)*x2length+b] = sink;  
    } 
}