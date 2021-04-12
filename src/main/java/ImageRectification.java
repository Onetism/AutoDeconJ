

import ij.*;
import ij.gui.*;
import ij.plugin.*;
import ij.process.*;

public class ImageRectification implements PlugIn{

    @Override
    public void run(String arg0) {
        if (IJ.versionLessThan("1.32c"))
            return;
        int[] wList = WindowManager.getIDList();
        if (wList == null){
            IJ.noImage();
            return;
        }
        String[] titles = new String[wList.length];
        for (int i = 0; i < wList.length; i++){
            ImagePlus imp = WindowManager.getImage(wList[i]);
            if (imp != null)
                titles[i] = imp.getTitle();
            else
                titles[i] = "";
        }
        String titleImage = Prefs.get("iterativedeconvolve3d.rawImage", titles[0]);
        int imageChoice = 0;
        for (int i = 0; i < wList.length; i++){
            if(titleImage.equals(titles[i])){
                imageChoice = i;
                break;
            }
        } 
		double dxcenter = Prefs.get("iterativedeconvolve3d.xcenter", 1300.6);
		double dycenter = Prefs.get("iterativedeconvolve3d.ycenter", 1066.5); 
		double ddx = Prefs.get("iterativedeconvolve3d.dx", 22.86);  
		int nNnum = (int)Prefs.get("iterativedeconvolve3d.Nnum", 15); 
		int dxcutleft = (int)Prefs.get("iterativedeconvolve3d.xcutleft", 0);
		int dxcutright = (int)Prefs.get("iterativedeconvolve3d.cutright", 0); 
		int dycutup = (int) Prefs.get("iterativedeconvolve3d.cutup", 0);
        int dycutdown = (int) Prefs.get("iterativedeconvolve3d.cutdown", 0);
        
		GenericDialog gd = new GenericDialog("imagerectification", IJ.getInstance());
        gd.addChoice("RawImage",titles,titles[imageChoice]);
        gd.addStringField("Output title","RectImage",20);
       	gd.addNumericField("xCenter",dxcenter, 3);      	
       	gd.addNumericField("yCenter", dycenter, 3);
       	gd.addNumericField("dx", ddx,3);
       	gd.addNumericField("Nnum", nNnum,0);
       	gd.addNumericField("XcutLeft", dxcutleft,0);
       	gd.addNumericField("XcutRight", dxcutright,0);
       	gd.addNumericField("YcutUp", dycutup,0);
       	gd.addNumericField("YcutDown", dycutdown,0);
        gd.showDialog();        

        if (gd.wasCanceled())
            return;  
        String savepath = IJ.getDirectory("where to save");

        ImagePlus Raw_Image = WindowManager.getImage(wList[gd.getNextChoiceIndex()]);
        String titleOut = gd.getNextString();
        dxcenter = gd.getNextNumber();
        dycenter = gd.getNextNumber();
        ddx = gd.getNextNumber();
        nNnum = (int)gd.getNextNumber();
        dxcutleft = (int)gd.getNextNumber();
        dxcutright = (int)gd.getNextNumber();
        dycutup = (int)gd.getNextNumber();
        dycutdown = (int)gd.getNextNumber();   
        
		Prefs.set("iterativedeconvolve3d.rawImage", Raw_Image.getTitle());
		Prefs.set("iterativedeconvolve3d.xCenter", dxcenter);
		Prefs.set("iterativedeconvolve3d.yCenter", dycenter);		
		Prefs.set("iterativedeconvolve3d.dx", ddx);  
		Prefs.set("iterativedeconvolve3d.Nnum", nNnum); 
		Prefs.set("iterativedeconvolve3d.XcutLeft", dxcutleft);
		Prefs.set("iterativedeconvolve3d.XcutRight", dxcutright); 
		Prefs.set("iterativedeconvolve3d.YcutUp", dycutup); 
		Prefs.set("iterativedeconvolve3d.YcutDown", dycutdown); 

        ImageProcessor ip_Image = Raw_Image.getProcessor();
        if(ip_Image instanceof ColorProcessor){
           IJ.showMessage("RGB images are not currently supported.");
           return;
       } 
        IJ.showMessage("Read the Raw image data!");
        ImageStack stackY = Raw_Image.getStack();
        int bw = stackY.getWidth();
        int bh = stackY.getHeight();
        int bd = Raw_Image.getStackSize();
        float[][] dataYin = new float[bd][];
        if(ip_Image instanceof FloatProcessor){
            for (int i = 0; i < bd; i++){
                dataYin[i] = (float[])stackY.getProcessor(i+1).getPixels();
            }
        }else{
            for (int i = 0; i < bd; i++){
                dataYin[i] = (float[])stackY.getProcessor(i+1).convertToFloat().getPixels();
            }
        } 

		double[] image = new double[bh*bw];
		for(int i = 0;i<bh;i++){
			for(int j = 0; j<bw;j++){
				image[i*bw+j] = dataYin[0][i*bw+j];
			}
        }
        
        // double dy = ddx;
        int M = nNnum;
        int mdiff = (int) Math.floor((double) nNnum / 2);

        double[] xresample = new double[(int) (Math.ceil(dxcenter * M / ddx)
                                    + Math.floor(((double) bw - dxcenter - 1) * M / ddx))];
        double[] yresample = new double[(int) (Math.ceil(dycenter * M / ddx)
                                    + Math.floor(((double) bh - dycenter - 1) * M / ddx))];
        for(int i = 0; i<(int) (Math.ceil(dxcenter * M / ddx)); i++){
            xresample[(int) (Math.ceil(dxcenter * M / ddx))-i-1] = dxcenter+1-i*ddx/M;
        }
        for(int i = 0; i<Math.floor(((double) bw - dxcenter - 1) * M / ddx); i++){
            xresample[(int) (Math.ceil(dxcenter * M / ddx))+i] = dxcenter+1+(i+1)*ddx/M;
        }       
        for(int i = 0; i<(int) (Math.ceil(dycenter * M / ddx)); i++){
            yresample[(int) (Math.ceil(dycenter * M / ddx))-i-1] = dycenter+1-i*ddx/M;
        }
        for(int i = 0; i<Math.floor(((double) bh - dycenter - 1) * M / ddx); i++){
            yresample[(int) (Math.ceil(dycenter * M / ddx))+i] = dycenter+1+(i+1)*ddx/M;
        }      

        double[] x = new double[bh];
        double[] y = new double[bw];
        for(int i = 0;i<bh; i++){
            x[i] = i;
        }
        for(int i = 0; i<bw; i++){
            y[i] = i;
        }

        double[] xq = new double[xresample.length*yresample.length];
        double[] yq = new double[yresample.length*xresample.length];
        int xqcenterInit = 0;
        int yqcenterInit = 0;
        for(int i = 0;i<yresample.length; i++){
            for(int j = 0; j<xresample.length; j++){
                yq[i*xresample.length+j] = yresample[i];
                xq[i*xresample.length+j] = xresample[j];
                if((xqcenterInit==0) && Math.abs(xresample[j]-dxcenter-1)<1e-8){
                    xqcenterInit = j;
                }
            }
            if((yqcenterInit==0) && Math.abs(yresample[i]-dycenter-1)<1e-8){
                yqcenterInit = i;
            }
        }

        xqcenterInit = xqcenterInit - mdiff;
        yqcenterInit = yqcenterInit - mdiff;

        int xqinit =  xqcenterInit - (int)(M*Math.floor((double)xqcenterInit/M))+M;
        int yqinit =  yqcenterInit - (int)(M*Math.floor((double)yqcenterInit/M))+M; 

        // int xqend = (int) (M * Math.floor((double) (xresample.length - xqinit+1) / M));
        // int yqend = (int) (M * Math.floor((double) (yresample.length - yqinit+1) / M));

        double[] xresampleQ = new double[xresample.length - xqinit];
        double[] yresampleQ = new double[yresample.length - yqinit];
        for(int i = 0;i<xresampleQ.length; i++){
            xresampleQ[i] = xresample[i+xqinit]-1;
        }
        for(int i = 0;i<yresampleQ.length; i++){
            yresampleQ[i] = yresample[i+yqinit]-1;
        }
        IJ.showMessage("start interpolating !");
        double[] img_resample = interp2d(x, y, image, bh, bw, yresampleQ, xresampleQ, yresampleQ.length, xresampleQ.length);
        int xcropsize  = (int) (M * Math.floor((double) (xresampleQ.length - xqinit) / M));
        int ycropsize  = (int) (M * Math.floor((double) (yresampleQ.length - yqinit) / M));

        double[] img_resample_crop = new double[xcropsize*ycropsize];
        for(int i = 0; i<ycropsize; i++){
            for(int j = 0; j<xcropsize; j++){
                img_resample_crop[i*xcropsize+j] = img_resample[i*xresampleQ.length+j];
            }
        }
        IJ.showMessage("interpolating  completely!");
        double xsizeML = (double)xcropsize/M;
        double ysizeML = (double)ycropsize/M;
        if ((dxcutleft+dxcutright-xsizeML)>1e10){
            IJ.showMessage("X-cut range is larger than the x-size of image");
        }
        if ((dycutup+dycutdown-ysizeML)>1e10){
            IJ.showMessage("Y-cut range is larger than the y-size of image");
        }
        int[] xrange = new int[(int) Math.ceil(xsizeML - dxcutright - dxcutleft)];
        int[] yrange = new int[(int) Math.ceil(ysizeML - dycutdown - dycutup)];
        for(int i = 0;i<xrange.length; i++){
            xrange[i] = dxcutleft+1+i;
        }
        for(int i = 0;i<yrange.length; i++){
            yrange[i] = dycutup+1+i;
        }
        int xcrop2size = xrange[xrange.length - 1] * M - (xrange[0] - 1) * M ;
        int ycrop2size = yrange[yrange.length - 1] * M - (yrange[0] - 1) * M ;

        double[] Img_rect = new double[xcrop2size*ycrop2size];
        double Img_rectMax = 0;
        for(int i =0 ;i< ycrop2size; i++){
            for(int j = 0; j<xcrop2size; j++){
                Img_rect[i*xcrop2size+j] = img_resample_crop[((xrange[0]-1)*M+i)*xcropsize+((yrange[0]-1)*M+j)];
                if(Img_rect[i*xcrop2size+j] > Img_rectMax){
                    Img_rectMax = Img_rect[i*xcrop2size+j];
                }
            }
        }
        for(int i =0 ;i< ycrop2size; i++){
            for(int j = 0; j<xcrop2size; j++){
                Img_rect[i*xcrop2size+j] = (int)(Img_rect[i*xcrop2size+j]/Img_rectMax*65535);
            }
        }        
        ImagePlus imageOutput = null;
        ImageStack stackOutput = new ImageStack(xcrop2size,ycrop2size);;
        
        ImageProcessor ip = new ShortProcessor(xcrop2size,ycrop2size);
        short[] px = (short[])ip.getPixels();
        for (int j = 0; j < ycrop2size; j++){
            for (int i = 0; i < xcrop2size; i++){
                px[i + xcrop2size*j] = (short) Img_rect[i + xcrop2size * j];
            }
        }
        ip.setMinAndMax(0,0);
        stackOutput.addSlice(null,ip);

        imageOutput = new ImagePlus(titleOut,stackOutput);
        imageOutput.show();

        IJ.saveAs(imageOutput, "tif", savepath+titleOut);
        IJ.showMessage("ImageRectification  completely!");

    }
	
    double[] interp2d(double[] x, double[] y, double[] z, int m, int n, double[] a, double[] b, int asize,int bsize)
    {
        double pointa = 0;
        double pointb = 0;
        double w1, w2, w;
        int nu_val = -9999;
        int tempi = 0;
        int tempj =0;
        double[] out_result = new double[asize*bsize];
        for (int i = 0; i < asize; ++i)
        {
            for (int j = 0; j < bsize; ++j)
            {
                pointa = a[i];
                pointb = b[j];
                for (int p = 0; p < m-1; ++p)
                {
                    if (pointa < x[0]||pointa>x[m-1])
                    {
                        tempi = nu_val;
                        break;
                    }
                    else if (pointa == x[m - 1])
                    {
                        tempi = m - 1;
                        break;
                    }
                    else if(pointa >= x[p] && pointa < x[p+1])
                    {
                        tempi = p;
                        break;
                    }
                }
                for (int q = 0; q < n - 1; ++q)
                {
                    if (pointb< y[0] || pointb>y[n- 1])
                    {
                        tempj = nu_val;
                        break;
                    }
                    else if (pointb == y[n - 1])
                    {
                        tempj = n - 1;
                        break;
                    }
                    else if (pointb >= y[q] && pointb < y[q + 1])
                    {
                        tempj = q;
                        break;
                    }
                }
                if (tempj == nu_val || tempi == nu_val)
                {
                    out_result[i*bsize + j] = nu_val;
                }
                else
                {         
                    if (x[tempi] == pointa)
                    {
                        w1 = z[tempi*n + tempj];
                        w2 = z[(tempi+1)*n + tempj];
        
                        if (y[tempj] == pointb)
                        {
                            w = w1;
                        }
                        else
                        {
                            w = inter_linear(y[tempj], w1, y[tempj + 1], w2, pointb);
                        }
                    }
                    else
                    {
                        w1 = inter_linear(x[tempi], z[tempi*n + tempj], x[tempi + 1], z[(tempi + 1)*n + tempj], pointa);
                        w2 = inter_linear(x[tempi], z[tempi*n + tempj + 1], x[tempi + 1], z[(tempi + 1)*n + tempj + 1], pointa); 
                        if (y[tempj] == pointb)
                        {
                            w = w1;
                        }
                        else
                        {
                            w = inter_linear(y[tempj], w1, y[tempj + 1], w2, pointb);
                        }
                    }
                    out_result[i*bsize + j] = w;       
                }    
            }
        }
        return out_result;
    }
    
    double inter_linear(double x0, double y0, double x1, double y1, double x)
    {
        double a0, a1, y;
        a0 = (x - x1) / (x0 - x1);
        a1 = (x - x0) / (x1 - x0);
        y = a0*y0 + a1*y1;
        return y;
    }
	
} 