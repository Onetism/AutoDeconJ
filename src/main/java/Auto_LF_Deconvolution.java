
/*
 * @Author: your name
 * @Date: 2021-01-21 11:36:58
 * @LastEditTime: 2022-08-23 23:19:12
 * @LastEditors: onetism onetism@163.com
 * @Description: In User Settings Edit
 * @FilePath: \LightFieldMicroscopy_ImageJPlugin\src\Auto_LF_Deconvolution.java
 */

/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */
import ij.*;

import ij.plugin.*;
import ij.process.*;
import java.lang.Math;
import java.text.DecimalFormat;
import java.io.*;

import javax.swing.*;
import javax.swing.border.TitledBorder;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.Color;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

import com.jmatio.io.MatFileReader;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdevice;
import jcuda.driver.JCudaDriver;
import static jcuda.driver.JCudaDriver.*;

import jcuda.*;
import jcuda.driver.*;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import jcuda.runtime.JCuda;

import static jcuda.runtime.JCuda.cudaFree;
import static jcuda.runtime.JCuda.cudaMalloc;
// import static jcuda.runtime.JCuda.cudaHostGetDevicePointer;
// import static jcuda.runtime.JCuda.cudaHostAlloc;
// import static jcuda.runtime.JCuda.cudaHostAllocMapped;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import java.util.Vector;

import org.jtransforms.dct.FloatDCT_2D;
/**
* This class is used to synchronize threads
*/
class Thread_syn{
	/**
     * Thread lock: ensure that data can only be accessed by one thread at a time
     */
	public static final Object lock = new Object();

	/**
     * The number of threads that have executed,which used to 
	 * determine the number of threads that have executed to the same place.
     */
	public static volatile int threadnum ;

	/**
     * Total number of threads in running
     */
	public static volatile int totalnum;

	/**
     * Recording the max value for each thread
     */
	public static volatile float[] max_value;
}



/**
* This is the main method, the PSF calculation and the deconvolution are all in there
*/

public class Auto_LF_Deconvolution implements PlugIn {

	/**
     * Thread lock: ensure that data can only be accessed by one thread at a time
     */	
	public static final Object lock = new Object();

	/**
     * Constants for Bessel function approximation.
     */	
	private static double[] t = new double[] { 1, -2.2499997, 1.2656208, -0.3163866, 0.0444479, -0.0039444, 0.0002100 };
	private static double[] p = new double[] { -.78539816, -.04166397, -.00003954, 0.00262573, -.00054125, -.00029333,.00013558 };
	private static double[] f = new double[] { .79788456, -0.00000077, -.00552740, -.00009512, 0.00137237, -0.00072805,0.00014476 };

	/**
     * The size of each dimension of the PSF
     */	
	volatile static int size_H1;
	volatile static int size_H2;
	volatile static int size_H3;
	volatile static int size_H4;
	volatile static int size_H5;

	/**
     * A window for information output
     */
	static JTextArea output;


	@Override
	public void run(String arg) {

		guiSet();
		while (true) {
			boolean run_flag = Prefs.get("Auto_LF_Deconvolution.run", false);
			if (run_flag) {
				output.append("Start Running!" + "\n");
				output.setCaretPosition(output.getDocument().getLength());
				start_run();
				run_flag = false;
				Prefs.set("Auto_LF_Deconvolution.run", false);
			}
		}
	}

	/**
     * Calculate the Bessel function approximation.
	 * 
	 * @param number the number to calculate
     * @return The Bessel function value of a number
     */	
	private double J0(double number) {
		double x = number;
		if (x < 0){
			x *= -1;
		}	
		double r = 0;
		if (x <= 3) {
			double y = x * x / 9;
			r = t[0] + y * (t[1] + y * (t[2] + y * (t[3] + y * (t[4] + y * (t[5] + y * t[6])))));
		} else {
			double y = 3 / x;
			double theta0 = x + p[0] + y * (p[1] + y * (p[2] + y * (p[3] + y * (p[4] + y * (p[5] + y * p[6])))));
			double f0 = f[0] + y * (f[1] + y * (f[2] + y * (f[3] + y * (f[4] + y * (f[5] + y * f[6])))));
			r = Math.sqrt(1 / x) * f0 * Math.cos(theta0);
		}
		return r;
	}
	
	/**
     * Start running 
     */	
	private void start_run() {

		if (IJ.versionLessThan("1.32c")) {
			return;
		}

		// Try to initialize the CUDA driver API
		JCudaDriver.cuInit(0);

		// Obtain the number of devices
		int deviceCountArray[] = { 0 };
		JCudaDriver.cuDeviceGetCount(deviceCountArray);
		int deviceCount = deviceCountArray[0];

		// Total memory for each cuda
		long total_cudamem = 0;

		// Record the available CUDA label
		Vector<Integer> avaliable_cuda = new Vector<Integer>();

		Vector<CUcontext> cuda_context = new Vector<CUcontext>();

		// Record the thread for computing psf
		Vector<Psfcompute_Thread> psfcomput_thread = new Vector<Psfcompute_Thread>();

		// cuInit(0);
		// Search for available CUDA,expcept for consuming memory more than 400M
		for(int i = 0; i<deviceCount; i++){
			
			CUdevice device = new CUdevice();
			cuDeviceGet(device, i);
			CUcontext context = new CUcontext();
			cuCtxCreate(context, 0, device);

			long []available_mem={0},total={0};
			JCuda.cudaMemGetInfo(available_mem,total);
			long free_mem = total[0] - available_mem[0];
			total_cudamem=total[0];
			cuCtxDestroy(context);
			if(free_mem < total_cudamem/2){
				avaliable_cuda.add(i);
			}
		}

		// // Whether P2P is supported between available CUDA
		// for(int i = 0; i< avaliable_cuda.size();i++){
		// 	int result = JCuda.cudaSetDevice(avaliable_cuda.get(i));
		// 	for (int j = i+1; j < avaliable_cuda.size(); j++) {
		// 		int access[] = {0};
		// 		JCuda.cudaDeviceCanAccessPeer(access, avaliable_cuda.get(i), avaliable_cuda.get(j));
		// 		if (access[0] == 1) {
		// 			result = JCuda.cudaDeviceEnablePeerAccess(avaliable_cuda.get(j), 0 );
		// 			if (result != CUresult.CUDA_SUCCESS){
		// 				output.append("Peer cannot from GPU"+avaliable_cuda.get(i) +"-> GPU" +avaliable_cuda.get(j)+"\n");
		// 				output.setCaretPosition(output.getDocument().getLength());
		// 			}else{
		// 				output.append("Peer access from GPU"+avaliable_cuda.get(i) +"-> GPU" +avaliable_cuda.get(j)+"\n");
		// 				output.setCaretPosition(output.getDocument().getLength());
		// 			}
		// 			result = JCuda.cudaSetDevice(avaliable_cuda.get(j));
		// 			if (result != CUresult.CUDA_SUCCESS){
		// 				output.append("cudaSetDevice erro!\n");
		// 				output.setCaretPosition(output.getDocument().getLength());
		// 			}
		// 			result = JCuda.cudaDeviceEnablePeerAccess(avaliable_cuda.get(i), 0 );
		// 			if (result != CUresult.CUDA_SUCCESS){
		// 				output.append("Peer cannot from GPU"+avaliable_cuda.get(j) +"-> GPU" +avaliable_cuda.get(i)+"\n");
		// 				output.setCaretPosition(output.getDocument().getLength());
		// 			}else{
		// 				output.append("Peer access from GPU"+avaliable_cuda.get(j) +"-> GPU" +avaliable_cuda.get(i)+"\n");
		// 				output.setCaretPosition(output.getDocument().getLength());
		// 			}
		// 			result = JCuda.cudaSetDevice(avaliable_cuda.get(i));
		// 			if (result != CUresult.CUDA_SUCCESS){
		// 				output.append("cudaSetDevice erro!\n");
		// 				output.setCaretPosition(output.getDocument().getLength());
		// 			}
		// 		}
		// 	}
		// }

		// // Destroy the cuda context
		// for(int i = 0; i< avaliable_cuda.size();i++){
		// 	CUdevice device = new CUdevice();
		// 	cuDeviceGet(device, avaliable_cuda.get(i));
		// 	CUcontext context = new CUcontext();
		// 	cuCtxCreate(context, 0, device);

		// 	cuCtxDestroy(context);
		// }
		JCuda.setExceptionsEnabled(true);

		boolean usepsfmat = Prefs.get("Auto_LF_Deconvolution.usepsf_from_matlab", false);
		String imagepath = Prefs.get("Auto_LF_Deconvolution.imagePath", null);
		String psfmat_path = Prefs.get("Auto_LF_Deconvolution.psfPath", null);
		boolean isUse_psffile = Prefs.get("Auto_LF_Deconvolution.isUse_psffile", false);
		String psf_filepath = Prefs.get("Auto_LF_Deconvolution.PSFfilePath", null);
		int ngpu_num = (int) Prefs.get("Auto_LF_Deconvolution.ngpu_num", 0);

		float[] psf_H = null;
		float[] psf_Ht = null;

		if (usepsfmat) {
			MatFileReader matread = null;
			try {
				output.append("Read psf from .mat!" + "\n");
				matread = new MatFileReader(psfmat_path);
			} catch (IOException e) {
				e.printStackTrace();
			}

			MLArray mlArrayH = matread.getMLArray("H");
			MLArray mlArrayHt = matread.getMLArray("Ht");

			int[] dimensions = mlArrayH.getDimensions();
			size_H1 = dimensions[0];
			size_H2 = dimensions[1];
			size_H3 = dimensions[2];
			size_H4 = dimensions[3];
			size_H5 = dimensions[4];
			psf_H = new float[size_H1 * size_H2 * size_H3 * size_H4 * size_H5];
			psf_Ht = new float[size_H1 * size_H2 * size_H3 * size_H4 * size_H5];
			MLDouble d_H = (MLDouble) mlArrayH;

			double[][] tempH = d_H.getArray();
			MLDouble d_Ht = (MLDouble) mlArrayHt;
			double[][] tempHt = d_Ht.getArray();

			for (int z = 0; z < size_H5; z++) {
				for (int x = 0; x < size_H4; x++) {
					for (int y = 0; y < size_H3; y++) {
						for (int i = 0; i < size_H1; i++) {
							for (int j = 0; j < size_H2; j++) {
								psf_H[z * size_H1 * size_H2 * size_H3 * size_H4 + x * size_H1 * size_H2 * size_H3
										+ y * size_H1 * size_H2 + i * size_H2
										+ j] = (float) tempH[i][z * size_H4 * size_H3 * size_H2 + x * size_H3 * size_H2
												+ y * size_H2 + j];
								psf_Ht[z * size_H1 * size_H2 * size_H3 * size_H4 + x * size_H1 * size_H2 * size_H3
										+ y * size_H1 * size_H2 + i * size_H2
										+ j] = (float) tempHt[i][z * size_H4 * size_H3 * size_H2 + x * size_H3 * size_H2
												+ y * size_H2 + j];
							}
						}
					}
				}
			}
			mlArrayH = null;
			mlArrayHt = null;
			d_H = null;
			d_Ht = null;
			tempH = null;
			tempHt = null;
			output.append(
					"Psf size:" + size_H1 + " " + size_H2 + " " + size_H3 + " " + size_H4 + " " + size_H5 + " " + "\n");
			output.setCaretPosition(output.getDocument().getLength());
		} else if (isUse_psffile) {
			ImagePlus imp_PSF = IJ.openImage(psf_filepath);
			int[] size = imp_PSF.getDimensions();
			size_H1 = size[0];
			size_H2 = size[1];
			size_H3 = size[2];
			size_H4 = size[3];
			size_H5 = size[4] / 2;

			ImageStack stack_PSF = imp_PSF.getStack();
			int psfw = stack_PSF.getWidth();
			int psfh = stack_PSF.getHeight();
			int psfz = imp_PSF.getStackSize();
			float[][] tempH = new float[psfz / 2][psfh * psfw];
			float[][] tempHt = new float[psfz / 2][psfh * psfw];
			for(int i = 0;i<psfz/2;i++){
				tempH[i] = (float[]) stack_PSF.getProcessor(i+1).getPixels();
				tempHt[i] = (float[]) stack_PSF.getProcessor(size_H3*size_H4*size_H5+i+1).getPixels();
			}
			psf_H = new float[psfz/2*psfh*psfw];
			psf_Ht = new float[psfz/2*psfh*psfw];
			for(int z = 0;z<size_H5;z++){
				for(int y = 0;y<size_H3;y++){
					for(int x =0;x<size_H4;x++){
						for(int i = 0;i<psfh;i++){
							for(int j = 0;j<psfw;j++){
								psf_H[z*psfw*psfh*size_H3*size_H4 + y*size_H3*psfw*psfh
								+x*psfw*psfh+i *psfw+j]
								= tempH[z*size_H3*size_H4+y*size_H3+x][i*psfw+j] ;
								psf_Ht[z*psfw*psfh*size_H3*size_H4 + y*size_H3*psfw*psfh
								+x*psfw*psfh+i *psfw+j]
								= tempHt[z*size_H3*size_H4+y*size_H3+x][i*psfw+j] ;
							}
						}

					}
				}
			}

		} else {
			long startTime = System.currentTimeMillis();

			output.append("Compute the psf_H!\n");
			output.setCaretPosition(output.getDocument().getLength());

			//predict the memory need for psf
			double n = Prefs.get("Auto_LF_Deconvolution.dn", 1.0);
			int Nnum = (int) Prefs.get("Auto_LF_Deconvolution.nNnum", 15);
			int OSR = (int) Prefs.get("Auto_LF_Deconvolution.nOSR", 3);
			double M = Prefs.get("Auto_LF_Deconvolution.dM", 40);
			double NA = Prefs.get("Auto_LF_Deconvolution.dna", 0.95);
			double MLPitch = Prefs.get("Auto_LF_Deconvolution.dMLPitch", 150) * 1e-6;
			double lambda =  Prefs.get("Auto_LF_Deconvolution.nlambda", 520) * 1e-9;
			double zmax = Prefs.get("Auto_LF_Deconvolution.dzmax", 0.0) * 1e-6;
			double zmin = Prefs.get("Auto_LF_Deconvolution.dzmin", -2.0) * 1e-6;
			double zspacing =  Prefs.get("Auto_LF_Deconvolution.dzspacing", 2.0)* 1e-6;
	
			double k = 2 * Math.PI * n / lambda; // the number of wave
			double alpha = Math.asin(NA / n);
			double ftl = 200e-3;
			double fobj = ftl / M;
	
			if (Math.floorMod(Nnum, 2) == 0) {
				output.append("Nnum should be an odd number." + "\n");
			}
	
			double pixelPitch = MLPitch / Nnum;
			double p3max = Math.max(Math.abs(zmax), Math.abs(zmin));
			double[] x1testspace = new double[Nnum * OSR * 50];
			for (int i = 0; i < x1testspace.length; i++) {
				x1testspace[i] = i * pixelPitch / OSR;
			}
			double[] psf_line = new double[x1testspace.length];
	
			double psflineMax = 0;
			for (int i = 0; i < x1testspace.length; i++) {
				double x1 = x1testspace[i];
				double x2 = 0;
				double xL2normsq = Math.sqrt(Math.pow(x1, 2) + Math.pow(x2, 2)) / M;
				double v = k * xL2normsq * Math.sin(alpha);
				double u = 4 * k * p3max * Math.pow(Math.sin(alpha / 2), 2);
				double Koi = M / Math.pow(fobj * lambda, 2);
				double ku1 = -1 * u / (4 * Math.pow(Math.sin(alpha / 2), 2));
				double cosk = 0;
				double sink = 0;
	
				
				double alphaspace = alpha / 10240;
				for (int j = 0; j < 10240; j++) {
					double ku2 = -1 * u * Math.pow(Math.sin(j * alphaspace / 2), 2)
							/ (2 * Math.pow(Math.sin(alpha / 2), 2));
					double ku3 = Koi * Math.sqrt(Math.cos(j * alphaspace)) * (1 + Math.cos(j * alphaspace))
							* J0(Math.sin(j * alphaspace) * v / Math.sin(alpha)) * Math.sin(j * alphaspace) * alphaspace;
					cosk += ku3 * Math.cos(ku2 + ku1); // real part in complex
					sink += ku3 * Math.sin(ku2 + ku1); // imaginary part in complex
				}
				double tempcosk = Math.pow(cosk, 2) - Math.pow(sink, 2);
				double tempsink = 2 * cosk * sink;
				psf_line[i] = Math.sqrt(Math.pow(tempcosk, 2) + Math.pow(tempsink, 2));
				if (psflineMax < psf_line[i]) {
					psflineMax = psf_line[i]; // get the max value in psf_line
				}
			}
			int outarea = 0; // 0:have out the max area others:don't out the max area
			for (int i = 0; i < x1testspace.length; i++) {
				psf_line[i] = psf_line[i] / psflineMax;
				if ((psf_line[i] < 0.04) && (outarea == 0)) {
					outarea = i;
				}
			}
			psf_line = null;
			x1testspace = null;
			int numpts = (int)((zmax - zmin) / zspacing + 1);
			int imgsize_ref = (int) Math.ceil((double) (outarea + 1) / (OSR * Nnum));
			int img_halfwidth = Math.max(Nnum * (imgsize_ref + 1), 2 * Nnum);
			if (outarea == 0) {
				IJ.showMessage("Estimated PSF size exceeds the limit" + "\n");
				return;
			}
		
			//the memory need for psf
			long psf_wave_mem = (long)Math.pow(img_halfwidth*OSR*2,2)*numpts*2*Sizeof.FLOAT;
			long mlarry_mem = (long)Math.pow(img_halfwidth*OSR*2,2)*2*4*Sizeof.FLOAT;
			long psf_mem = (long)Math.pow(2*Nnum*imgsize_ref,2)*Nnum*Nnum*numpts*Sizeof.FLOAT;
			
			//the memory need for psfht
			long ht_fft_mem = (long)Math.pow(3*Nnum*imgsize_ref, 2)*4*Sizeof.FLOAT;
			long ht_mem = psf_mem;
			long temp_result_mem = (long)Math.pow(2*Nnum*imgsize_ref, 2)*numpts*Sizeof.FLOAT; //a little large

			
			long require_mem = (long)(temp_result_mem+ht_mem+ht_fft_mem+psf_wave_mem+mlarry_mem+psf_mem)/1024/1024; //11800 is the reserved memory
			total_cudamem = total_cudamem/1024/1024;

			int cudasNeed = (int)Math.ceil((double)require_mem/(double)total_cudamem);

			if(ngpu_num!=0 ){
				if (ngpu_num>=cudasNeed)
					cudasNeed = ngpu_num;
				else{
					Auto_LF_Deconvolution.output.append( cudasNeed +" GPUs are need, but the setting GPUs "+ ngpu_num  + "  is not enough!\n");
					Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());		
					return;			
				}
			}

			if(cudasNeed > avaliable_cuda.size()){
				// IJ.showMessage("The number of GPU is not enough!" + "\n");
				Auto_LF_Deconvolution.output.append( cudasNeed +" GPUs are need, but there are only  "+ avaliable_cuda.size()  + "  in this system!\n");
				Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
				return;
			}
			int numpts_spacing =(int)((double)numpts/cudasNeed+0.5);

			for(int i = 0; i<cudasNeed; i++){
			
				CUdevice device = new CUdevice();
				cuDeviceGet(device, avaliable_cuda.get(i));
				CUcontext context = new CUcontext();
				cuCtxCreate(context, 0, device);
				cuda_context.add(context);
			}
			//Thread synchronization Settings
			Thread_syn.threadnum = cudasNeed;	
			Thread_syn.totalnum = cudasNeed;	
			Thread_syn.max_value = new float[cudasNeed];	

			double temp_zmin,temp_zmax;
			
			//The PSF is calculated in layers as needed
			for(int i = 0; i< cudasNeed; i++){
				if(i == 0){
					temp_zmin = (zmin);
					temp_zmax = zmin+(numpts_spacing*(i+1)-1)*zspacing;
				}
				else if(i!=cudasNeed-1 && i!=0){
					temp_zmin = zmin+(numpts_spacing*i)*zspacing;
					temp_zmax = zmin+(numpts_spacing*(i+1)-1)*zspacing;
				}
				else{
					temp_zmin = zmin+(numpts_spacing*i)*zspacing;
					temp_zmax = zmax;	
				}
				
				Psfcompute_Thread psf_thread =  new Psfcompute_Thread(p3max,outarea,temp_zmin, temp_zmax,avaliable_cuda.get(i), cuda_context.get(i));
				psf_thread.start();
				psfcomput_thread.add(psf_thread);
			}		
					
			for(Thread test:psfcomput_thread){
				try{
					test.join();
				}catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			
			output.append(
					"Psf size:" + size_H1 + " " + size_H2 + " " + size_H3 + " " + size_H4 + " " + size_H5 + " " + "\n");
			output.setCaretPosition(output.getDocument().getLength());
			
			long endTime = System.currentTimeMillis();
			output.append("Compute_psf take " + (float) (endTime - startTime) / 1000 + "s" + "\n");
			output.setCaretPosition(output.getDocument().getLength());
		}
		// File file = new File(imagepath);
		// File[] fileList = file.listFiles();

		// for(int filenum = 0; filenum < fileList.length;filenum++){

			// if(fileList[filenum].getName().endsWith(".tiff") == false
			// && fileList[filenum].getName().endsWith(".tif") == false){
			// 	continue;
			// }	
			// String filepath = fileList[filenum].getAbsoluteFile().getName();
			// String filename = null;
			// if(fileList[filenum].getName().endsWith(".tiff") == true){
			// 	filename = filepath.substring(0, filepath.length()-5);
			// }
			// else{
			// 	filename = filepath.substring(0, filepath.length()-4);
			// }
			
			// ImagePlus imp_Image = IJ.openImage(fileList[filenum].getPath());
			ImagePlus imp_Image = IJ.openImage(imagepath);

			output.append("Read light filed data : "+ imagepath + "\n");
			// output.append(filenum+" | "+fileList.length +"  "+"Read " + filename+ " light filed data!"+ "\n");
			output.setCaretPosition(output.getDocument().getLength());

			ImageProcessor ip_Image = imp_Image.getProcessor();
			if (ip_Image instanceof ColorProcessor) {
				IJ.showMessage("RGB images are not currently supported.");
				return;
			}
	
			ImageStack stackY = imp_Image.getStack();
			int bw = stackY.getWidth();
			int bh = stackY.getHeight();
			int bd = imp_Image.getStackSize();
			float[][] dataYin = new float[bd][];
			if (ip_Image instanceof FloatProcessor) {
				for (int i = 0; i < bd; i++) {
					dataYin[i] = (float[]) stackY.getProcessor(i + 1).getPixels();
				}
			} else {
				for (int i = 0; i < bd; i++) {
					dataYin[i] = (float[]) stackY.getProcessor(i + 1).convertToFloat().getPixels();
				}
			}
	
			float[] image = new float[bh * bw];
			for (int i = 0; i < bh; i++) {
				for (int j = 0; j < bw; j++) {
					image[i * bw + j] = dataYin[0][i * bw + j];
				}
			}
			dataYin = null;

			//Thread synchronization Settings
			Thread_syn.threadnum = psfcomput_thread.size();
			Thread_syn.totalnum = psfcomput_thread.size();
			Thread_syn.max_value = new float[psfcomput_thread.size()];

			AutoDconThread autodeconJ = new AutoDconThread(psfcomput_thread,image,new int[]{bh,bw},psfcomput_thread.size(), cuda_context);

			Vector<Thread> deconJvector = new Vector<Thread>();
			for(int i = 0; i< psfcomput_thread.size();i++){
				Thread test = new Thread(autodeconJ);
				test.start();
				deconJvector.add(test);
			}
			for(Thread test:deconJvector){
				try{
					test.join();
				}catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
			// Destroy the cuda context
			for(int i = 0; i< cuda_context.size();i++){
				cuCtxDestroy(cuda_context.get(i));
			}			
			// int wait = 0;
		// }


		output.append("Complete! " + "\n");
		output.setCaretPosition(output.getDocument().getLength());

	}

	/**
	* This class is used to deconvolution
	*/
	class AutoDconThread implements Runnable{

		/**
		 * Thread lock: ensure that data can only be accessed by one thread at a time
		 */	
		// private final Object lock = new Object();

		private float[][] host_forward_resultP;

		/**
		 * Record the results of the forward projection for each thread
		 */	
		private Pointer[] forward_resultP;	

		/**
		 * Record the results of the iterarion for each thread
		 */	
		private Pointer[] result_xguessP;			

		/**
		 * input the psf  psfht psfsize
		 */	
		private Vector<Psfcompute_Thread> psfH;	

		/**
		 * the data for inpute light-field iamge
		 */	
		private float[] inputImage;	

		/**
		 * the max thread need to open
		 */	
		private volatile int threadNum;	

		/**
		 * the size of input image
		 */			
		private int[] image_size;	

		/**
		 * the psfsupportdiameter for iamge quality
		 */			
		private double npfsupportdiameter;	

		/**
		 * the max iteration for deconvolution
		 */			
		private volatile int maxIter;

		boolean  findbest;

		private Vector<CUcontext> cuda_context;


		AutoDconThread(Vector<Psfcompute_Thread> thread,float[] image,int[] size,int num, Vector<CUcontext> context){
			psfH = thread;
			inputImage = image;
			image_size = size;
			threadNum = num;
			findbest = false;
			result_xguessP = new Pointer[threadNum];
			forward_resultP = new Pointer[threadNum];
			
			npfsupportdiameter = Prefs.get("Auto_LF_Deconvolution.npsfsupportdiameter", 11.9);
			maxIter = (int) Prefs.get("Auto_LF_Deconvolution.nIter", 20);
			host_forward_resultP = new float[num][size[0]*size[1]];
			cuda_context = context;
		}

		@Override
		public void run() {
			if(threadNum<0)
				return;
			int current_thread = 0;

			//each thread started, the number of threads needed is reduced 
			synchronized(this){
				threadNum--;
				current_thread = threadNum;
				// JCuda.cudaSetDevice(psfH.get(current_thread).device_num);
				// CUdevice device = new CUdevice();
				// cuDeviceGet(device, psfH.get(current_thread).device_num);
				cuCtxSetCurrent(cuda_context.get(current_thread));
			}

			int []size = psfH.get(current_thread).psf_pointer_result.size;
			KernelLauncher[] kernelLauncher = new KernelLauncher[7];
			kernelLauncher[0] = Kernelsetup.cudaFilesetup("getHnew.ptx", "getHnew");
			kernelLauncher[1] = Kernelsetup.cudaFilesetup("multiply.ptx", "multiply");
			kernelLauncher[2] = Kernelsetup.cudaFilesetup("add.ptx", "add");
			kernelLauncher[3] = Kernelsetup.cudaFilesetup("getprojection.ptx", "getprojection");	
			kernelLauncher[4] = Kernelsetup.cudaFilesetup("forwardADD.ptx", "forwardADD");
			kernelLauncher[5] = Kernelsetup.cudaFilesetup("xguessmax.ptx", "xguessmax");	
			kernelLauncher[6] = Kernelsetup.cudaFilesetup("tozero.ptx", "tozero");		
			
			float[] xguess = null;
			if(current_thread == 0){
				xguess = new float[image_size[0]*image_size[1]*size_H5];
			}

			Pointer device_inputimage = new Pointer();
			cudaMalloc(device_inputimage,image_size[0]*image_size[1]*Sizeof.FLOAT);
			cudaMemcpy(device_inputimage, Pointer.to(inputImage), image_size[0]*image_size[1]*Sizeof.FLOAT, cudaMemcpyHostToDevice);

			int back_size = image_size[0]*image_size[1]*size[4];
			int forward_size = image_size[0]*image_size[1];

			Pointer Htf = new Pointer();
			cudaMalloc(Htf, back_size*Sizeof.FLOAT);

			forward_resultP[current_thread] = new Pointer();
			// cudaHostAlloc(forward_resultP[current_thread], forward_size*Sizeof.FLOAT, cudaHostAllocMapped);

			// Pointer device_forward_resultP = new Pointer();
			// cudaHostGetDevicePointer(device_forward_resultP, forward_resultP[current_thread], 0);
			cudaMalloc(forward_resultP[current_thread], forward_size*Sizeof.FLOAT);

			backprojectionFFT(psfH.get(current_thread).Ht_pointer,device_inputimage,Htf,image_size,size,kernelLauncher);

			cudaFree(device_inputimage);

			result_xguessP[current_thread] = new Pointer();
			// Pointer device_xguess = new Pointer();
			cudaMalloc(result_xguessP[current_thread], back_size*Sizeof.FLOAT);
			cudaMemcpy(result_xguessP[current_thread], Htf, back_size*Sizeof.FLOAT, cudaMemcpyDeviceToDevice);

			// double[] tempback = new double[back_size];
			// cudaMemcpy(Pointer.to(tempback), device_xguess, back_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);

			// Pointer host_hxguess = new Pointer();
			// cudaHostAlloc(host_hxguess, forward_size*Sizeof.FLOAT, cudaHostAllocMapped);

			Pointer device_hxguess = new Pointer();
			cudaMalloc(device_hxguess, forward_size*Sizeof.FLOAT);
			// cudaHostGetDevicePointer(device_hxguess, host_hxguess, 0);

			Pointer device_hxguessback = new Pointer();
			cudaMalloc(device_hxguessback, back_size*Sizeof.FLOAT);

			ImagePlus imageOutput = null;
			ImageStack stackOutput = null;
			int output_w = image_size[1] - 0 * size[2];
			int output_h = image_size[0] - 0 * size[2];

			
			double[] dct_value = new double[maxIter];

			DecimalFormat df = new DecimalFormat("#.0000");
			long startTime = 0, endTime = 0;
			String savepath = Prefs.get("Auto_LF_Deconvolution.savePath", null);
			boolean perstore = Prefs.get("Auto_LF_Deconvolution.storperiter", false);
			String outputName = Prefs.get("Auto_LF_Deconvolution.outputName", "Iter");
			boolean is_Showper_result = Prefs.get("Auto_LF_Deconvolution.is_Showper_result", false);

			Pointer tempadd = new Pointer();
			cudaMalloc(tempadd, image_size[0]*image_size[1]*Sizeof.FLOAT);
			int resultSize =  (int) Math.ceil((double) ((long)image_size[0]*image_size[1]*size[4]) / 1024);

			Pointer tempresult = new Pointer();
			cudaMalloc(tempresult, resultSize*Sizeof.FLOAT);

			Pointer device_max = new Pointer();
			cudaMalloc(device_max, Sizeof.FLOAT);

			Pointer blocknum = new Pointer();
			cudaMalloc(blocknum, Sizeof.INT);

			Pointer temp_forward_resultP = new Pointer();
			cudaMalloc(temp_forward_resultP, forward_size*Sizeof.FLOAT);


			for(int iter =0; iter < maxIter; iter++){
				if (findbest)
					break;
				if(current_thread == 0)
					startTime = System.currentTimeMillis();

				 forwardprojectionFFT(psfH.get(current_thread).psf_pointer_result.pointer,
				 			result_xguessP[current_thread],forward_resultP[current_thread],image_size,size,kernelLauncher);
				cudaMemcpy(Pointer.to(host_forward_resultP[current_thread]), forward_resultP[current_thread], forward_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
				synchronized(Thread_syn.lock){
					Thread_syn.threadnum--;
				}

				//Thread synchronization, waiting for all threads to run here
				while(true){
					if(Thread_syn.threadnum == 0){
						break;
					}
					if(findbest){
						synchronized(Thread_syn.lock){
							Thread_syn.threadnum = Thread_syn.totalnum;
							break;
						}
					}
						
					// try {
					// 	Thread.sleep(20);
					// } catch (InterruptedException e) {
					// 	e.printStackTrace();
					// }	
				}
				if(Thread_syn.threadnum == Thread_syn.totalnum)
					break;
				// double[] tempv = new double[forward_size];
				// cudaMemcpy(Pointer.to(tempv), forward_resultP[current_thread], forward_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
		
				int blockSize = 1024;
				int gridSize =  (int) Math.ceil((double) ((long)image_size[0]*image_size[1]) / blockSize);
				kernelLauncher[4].setGridSize(gridSize, 1);
				kernelLauncher[4].setBlockSize(blockSize, 1, 1);
				kernelLauncher[4].call(forward_resultP[current_thread],device_hxguess,tempadd,image_size[0],image_size[1],0);

				//Sum the forward projections of each thread
				// Pointer temp_forward_resultP = new Pointer();
				// cudaMalloc(temp_forward_resultP, forward_size*Sizeof.FLOAT);
				// cudaHostGetDevicePointer(temp_forward_resultP, forward_resultP[i], 0);
				for(int i = 0;i< psfH.size() ; i++){
					cudaMemcpy(temp_forward_resultP, Pointer.to(host_forward_resultP[i]), forward_size*Sizeof.FLOAT, cudaMemcpyHostToDevice);
					kernelLauncher[4].call(temp_forward_resultP,device_hxguess,tempadd,image_size[0],image_size[1],1);				
				}
				
				float[] tempf = new float[forward_size];
				cudaMemcpy(Pointer.to(tempf), device_hxguess, forward_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
				// try {
				// 	Thread.currentThread().sleep(20);
				// } catch (InterruptedException e) {
				// 	e.printStackTrace();
				// }
			
				kernelLauncher[6].setGridSize(gridSize, 1);
				kernelLauncher[6].setBlockSize(blockSize, 1, 1);
				kernelLauncher[6].call(forward_resultP[current_thread],forward_size);

				kernelLauncher[6].setGridSize(gridSize, 1, 1);
				kernelLauncher[6].setBlockSize(blockSize, 1, 1);
				kernelLauncher[6].call(tempadd,forward_size);
			
				backprojectionFFT(psfH.get(current_thread).Ht_pointer,device_hxguess,device_hxguessback,image_size,size,kernelLauncher);
				
				// float[] tempb = new float[back_size];
				// cudaMemcpy(Pointer.to(tempb), device_hxguessback, back_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);

				// float[] temph = new float[forward_size];
				// cudaMemcpy(Pointer.to(temph), forward_resultP[current_thread], forward_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
				
				gridSize =  (int) Math.ceil((double) ((long)image_size[0]*image_size[1]*size[4]) / blockSize);

				// Pointer tempresult = new Pointer();
				// cudaMalloc(tempresult, gridSize*Sizeof.FLOAT);

				// Pointer device_max = new Pointer();
				// cudaMalloc(device_max, Sizeof.FLOAT);

				// Pointer blocknum = new Pointer();
				// cudaMalloc(blocknum, Sizeof.INT);
				cudaMemcpy(blocknum, Pointer.to(new int[]{0}), Sizeof.INT, cudaMemcpyHostToDevice);
				
				kernelLauncher[5].setGridSize(gridSize, 1);
				kernelLauncher[5].setBlockSize(blockSize, 1, 1);
				
				//Reduction to the maximum
				kernelLauncher[5].call(Htf,device_hxguessback,result_xguessP[current_thread],tempresult,
									device_max,blocknum,gridSize,(long)image_size[0]*image_size[1]*size[4]);

				float result[] = {0};
				cudaMemcpy(Pointer.to(result),device_max, Sizeof.FLOAT, cudaMemcpyDeviceToHost);


				Thread_syn.max_value[current_thread] = result[0];

				kernelLauncher[6].setGridSize(gridSize, 1);
				kernelLauncher[6].setBlockSize(blockSize, 1, 1);
				kernelLauncher[6].call(device_hxguessback,back_size);
				
				synchronized(Thread_syn.lock){
					Thread_syn.threadnum++;
				}
				while(true){
					if(Thread_syn.threadnum == Thread_syn.totalnum){
						break;
					}

					// try {
					// 	Thread.sleep(20);
					// } catch (InterruptedException e) {
					// 	e.printStackTrace();
					// }		
				}
				if(current_thread == 0){
					endTime = System.currentTimeMillis();
					Auto_LF_Deconvolution.output.append("Iter " + (iter+1) +" | "+maxIter +" take " + (float) (endTime - startTime) / 1000
									+ "s " + "\n");
					Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
				}

				float max_xguess = 0;
				for(int i = 0; i< Thread_syn.max_value.length; i++){
					if(max_xguess < Thread_syn.max_value[i]){
						max_xguess = Thread_syn.max_value[i];
					}
				}	

				if(current_thread == 0){
					float[] tempxguess = null;
					int offset = 0;
					for(int i = 0;i<psfH.size();i++){
						tempxguess = new float[image_size[0]*image_size[1]*psfH.get(i).psf_pointer_result.size[4]];
						cudaMemcpy(Pointer.to(tempxguess), result_xguessP[i], tempxguess.length*Sizeof.FLOAT, cudaMemcpyDeviceToHost);

						System.arraycopy(tempxguess, 0, xguess, offset, tempxguess.length);
						offset += tempxguess.length;
					}	
				
					// Auto_LF_Deconvolution.output.append("cudaMemcpy take " + (float) (endtime - starttime) / 1000+ "s " + "\n");
					stackOutput = new ImageStack(output_w, output_h);
					ImageStack fstackOutput = new ImageStack(output_w, output_h);

					for (int z = 0; z < size_H5; z++) {
						ImageProcessor ip = new ShortProcessor(output_w, output_h);
						ImageProcessor fip = new FloatProcessor(output_w, output_h);
						short[] px = (short[]) ip.getPixels();
						float[] fpx = (float[]) fip.getPixels();
						for (int i = 0; i < output_h; i++) {
							for (int j = 0; j < output_w; j++) {
								fpx[i * output_w + j] = (xguess[z*image_size[0]*image_size[1]+(i+0*size[2])* image_size[1]+j + 0 * size[2]]
										/ max_xguess );
								px[i * output_w + j] = (short)(fpx[i * output_w + j]*65535);
								
								fpx[i * output_w + j] = fpx[i * output_w + j]*128;
							}
						}
						ip.setMinAndMax(0, 0);
						stackOutput.addSlice(null, ip);
						fstackOutput.addSlice(null, fip);
					}

					imageOutput = new ImagePlus(outputName + "_" + String.valueOf(iter+1), stackOutput);
					if(is_Showper_result)
						imageOutput.show();
					if (perstore){
						IJ.saveAs(imageOutput, "tif", savepath + "/" + String.valueOf(iter+1) + "_" + outputName);
					}
					else{
						ZProjector z_projector = new ZProjector();
						ImagePlus maxZ_projection = z_projector.run(new ImagePlus("",fstackOutput), "max");
						ImageStack zTemp = maxZ_projection.getStack();
						float[] fz_pixels = (float[]) zTemp.getProcessor(1).getPixels();
						double[] dz_pixels = new double[output_w*output_h];
						for(int i = 0 ; i < output_h; i++){
							for(int j = 0; j < output_w; j++){
								dz_pixels[i * output_w + j] = fz_pixels[i * output_w + j];
							}
						}
	
						DCTEntropy dct_calculate = new DCTEntropy(output_w, output_h, dz_pixels.clone());
						dct_value[iter] = dct_calculate.compute((float)npfsupportdiameter);
						if(iter != 0 && dct_value[iter] - dct_value[iter-1] < 0){
							synchronized(Thread_syn.lock){
								findbest = true;
								Thread_syn.threadnum = Thread_syn.totalnum;
							}
							Auto_LF_Deconvolution.output.append( "The result best at " + (iter) + "\n");
							Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());	

							IJ.saveAs(imageOutput, "tif", savepath + "/" + String.valueOf(iter+1) +  "_" + outputName);
							break;
						}
					}

				}

			}
			cudaFree(temp_forward_resultP);
			cudaFree(device_max);
			cudaFree(tempresult);
			cudaFree(blocknum);
			xguess = null;
			return;	
		}
		private void forwardprojectionFFT(Pointer device_H,Pointer device_srcprojection,Pointer device_result, int[] projection_size,int[] size, KernelLauncher[] kernelLaunchers) {


			// float[] result = new float[projection_size[0] * projection_size[1]];		
			int[] xsize = new int[] { projection_size[0], projection_size[1] };
			int[] mmid = new int[] { (int) Math.floor((float) size[0] / 2), (int) Math.floor((float) size[1] / 2) };
			int[] temp_exsize = new int[] { xsize[0] + mmid[0], xsize[1] + mmid[1] };
			int exsize1 = (int) Math.min(Math.pow(2.0, Math.ceil(Math.log(temp_exsize[0]) / Math.log(2))),
					128 * Math.ceil((float) temp_exsize[0] / 128));
			int exsize2 = (int) Math.min(Math.pow(2.0, Math.ceil(Math.log(temp_exsize[1]) / Math.log(2))),
					128 * Math.ceil((float) temp_exsize[1] / 128));
			
			int blockSize = 32;
			int gridSize = (int) Math.ceil((float) Math.max(exsize1, exsize2) / blockSize);
	
			kernelLaunchers[0].setGridSize(gridSize, gridSize);
			kernelLaunchers[0].setBlockSize(blockSize, blockSize, 1);
	
			kernelLaunchers[1].setGridSize(gridSize, gridSize);
			kernelLaunchers[1].setBlockSize(blockSize, blockSize, 1);
	
			int gridSize3 = (int) Math.ceil((float) Math.max(projection_size[0], projection_size[1]) / blockSize);
			kernelLaunchers[2].setGridSize(gridSize3, gridSize3);
			kernelLaunchers[2].setBlockSize(blockSize, blockSize, 1);
	
			kernelLaunchers[3].setGridSize(gridSize, gridSize);
			kernelLaunchers[3].setBlockSize(blockSize, blockSize, 1);
	
			long fftsize = (long) exsize1 * exsize2 * 2 * Sizeof.FLOAT;
			Pointer device_projection = new Pointer();
			cudaMalloc(device_projection, fftsize);
	
			cufftHandle plan = new cufftHandle();
			JCufft.cufftPlan2d(plan, exsize1, exsize2, cufftType.CUFFT_C2C);
	
			Pointer device_Hnew = new Pointer();
			cudaMalloc(device_Hnew, fftsize);
	
			Pointer device_dst = new Pointer();
			cudaMalloc(device_dst, fftsize);
	
			Pointer device_tempresult = new Pointer();
			cudaMalloc(device_tempresult,  projection_size[0]*projection_size[1]*Sizeof.FLOAT);
			for (int cc = 0; cc < size[4]; cc++) {
				for (int aa = 0; aa < size[2]; aa++) {
					for (int bb = 0; bb < size[3]; bb++) {
						kernelLaunchers[3].call(device_srcprojection, device_projection, size[2], exsize1, exsize2, aa, bb, cc,
							projection_size[0], projection_size[1], 0);
						kernelLaunchers[3].call(device_srcprojection, device_projection, size[2], exsize1, exsize2, aa, bb,cc,
							projection_size[0], projection_size[1], 1);
	
						kernelLaunchers[0].call(device_H, device_Hnew, aa, bb, cc, mmid[0], mmid[1], exsize1, exsize2, size[0],
								size[1], size[2], size[3], size[4], 0);

						JCufft.cufftExecC2C(plan, device_Hnew, device_Hnew, JCufft.CUFFT_FORWARD);
						JCufft.cufftExecC2C(plan, device_projection, device_projection, JCufft.CUFFT_FORWARD);
						

						kernelLaunchers[1].call(device_projection, device_Hnew, device_dst, exsize1, exsize2);
						
						JCufft.cufftExecC2C(plan, device_dst, device_dst, JCufft.CUFFT_INVERSE);
						
						kernelLaunchers[2].call(device_dst,device_result,device_tempresult, aa, bb, cc, size[2], exsize1, exsize2, projection_size[0],
								projection_size[1], 2);
								
					}
				}
			}
			// cudaMemcpy(Pointer.to(result), device_result, result_size, cudaMemcpyDeviceToHost);
			JCufft.cufftDestroy(plan);
			cudaFree(device_projection);
			cudaFree(device_tempresult);
			// cudaFree(device_H);
			cudaFree(device_dst);
			cudaFree(device_Hnew);
			// cudaFree(device_result);
			// cudaFree(device_srcprojection);
			return ;
		}		
		private void backprojectionFFT(Pointer device_H,Pointer device_srcprojection,Pointer device_result,int[] projection_size, int[] size,KernelLauncher[] kernelLaunchers) {

			// float[] result = new float[projection_size[0] * projection_size[1] * size[4]];
	
			// int result_length = projection_size[0] * projection_size[1] * size[4];
			// int projection_length = projection_size[0] * projection_size[1];

			int[] xsize = new int[] { projection_size[0], projection_size[1] };
			int[] mmid = new int[] { (int) Math.floor((double) size[0] / 2), (int) Math.floor((double) size[1] / 2) };
			int[] temp_exsize = new int[] { xsize[0] + mmid[0], xsize[1] + mmid[1] };
			int exsize1 = (int) Math.min(Math.pow(2.0, Math.ceil(Math.log(temp_exsize[0]) / Math.log(2))),
					128 * Math.ceil((float) temp_exsize[0] / 128));
			int exsize2 = (int) Math.min(Math.pow(2.0, Math.ceil(Math.log(temp_exsize[1]) / Math.log(2))),
					128 * Math.ceil((float) temp_exsize[1] / 128));
			// int[] exsize = new int[]{exsize1,exsize2};
	
			int blockSize = 32;
			int gridSize = (int) Math.ceil((double) Math.max(exsize1, exsize2) / blockSize);
	
			kernelLaunchers[0].setGridSize(gridSize, gridSize);
			kernelLaunchers[0].setBlockSize(blockSize, blockSize, 1);
	
			kernelLaunchers[1].setGridSize(gridSize, gridSize);
			kernelLaunchers[1].setBlockSize(blockSize, blockSize, 1);
	
			int gridSize3 = (int) Math.ceil((double) Math.max(projection_size[0], projection_size[1]) / blockSize);
			
			kernelLaunchers[2].setGridSize(gridSize3, gridSize3);
			kernelLaunchers[2].setBlockSize(blockSize, blockSize, 1);
	
			kernelLaunchers[3].setGridSize(gridSize, gridSize);
			kernelLaunchers[3].setBlockSize(blockSize, blockSize, 1);
	
			long fftsize = (long) exsize1 * exsize2 * 2 * Sizeof.FLOAT;
			Pointer device_projection = new Pointer();
			cudaMalloc(device_projection, fftsize);
	
			cufftHandle plan = new cufftHandle();
			JCufft.cufftPlan2d(plan, exsize1, exsize2, cufftType.CUFFT_C2C);
	
			Pointer device_Hnew = new Pointer();
			cudaMalloc(device_Hnew, fftsize);
	
			Pointer device_dst = new Pointer();
			cudaMalloc(device_dst, fftsize);

			Pointer device_tempresult = new Pointer();
			cudaMalloc(device_tempresult,  projection_size[0]*projection_size[1]*size[4]*Sizeof.FLOAT);

			for (int cc = 0; cc < size[4]; cc++) {
				for (int aa = 0; aa < size[2]; aa++) {
					for (int bb = 0; bb < size[3]; bb++) {
					
						kernelLaunchers[3].call(device_srcprojection, device_projection, size[2], exsize1, exsize2, aa, bb, cc,
								projection_size[0], projection_size[1], 0);
						kernelLaunchers[3].call(device_srcprojection, device_projection, size[2], exsize1, exsize2, aa, bb, cc,
								projection_size[0], projection_size[1], 2);
		
						kernelLaunchers[0].call(device_H, device_Hnew, aa, bb, cc, mmid[0], mmid[1], exsize1, exsize2, size[0],
								size[1], size[2], size[3], size[4], 0);
		
						JCufft.cufftExecC2C(plan, device_Hnew, device_Hnew, JCufft.CUFFT_FORWARD);		
						JCufft.cufftExecC2C(plan, device_projection, device_projection, JCufft.CUFFT_FORWARD);

						kernelLaunchers[1].call(device_projection, device_Hnew, device_dst, exsize1, exsize2);
					
						JCufft.cufftExecC2C(plan, device_dst, device_dst, JCufft.CUFFT_INVERSE);
						kernelLaunchers[2].call(device_dst,device_result,device_tempresult,aa, bb, cc, size[2], exsize1, exsize2, projection_size[0],
							projection_size[1], 3);
					}
				}
			}
			// cudaMemcpy(Pointer.to(result), device_result, result_size, cudaMemcpyDeviceToHost);
			JCufft.cufftDestroy(plan);
			cudaFree(device_projection);
			cudaFree(device_tempresult);
			// cudaFree(device_H);
			cudaFree(device_dst);
			cudaFree(device_Hnew);
			// cudaFree(device_result);
			// cudaFree(device_srcprojection);
			return ;
		}

	}


	private void guiSet(){
		String os_name = System.getProperties().getProperty("os.name");
		String fonts = "Serif";
		int fontsize = 14;
		if (os_name.equals("Linux")) {
			fontsize = 14;
		} else {
			fontsize = 18;
		}
		
		JFrame.setDefaultLookAndFeelDecorated(true);
		JFrame frame = new JFrame("Auto_LF_Deconvolution");
		frame.setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
		frame.setSize(880, 830);
		frame.setLocationRelativeTo(null);
		frame.setResizable(true);
	
		JSplitPane frame_panel = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
		JSplitPane frame_panel2 = new JSplitPane(JSplitPane.VERTICAL_SPLIT);
		frame_panel.setContinuousLayout(true);
		frame_panel2.setContinuousLayout(true);
		frame_panel.setBottomComponent(frame_panel2);
		frame_panel.setDividerLocation(150);
		frame_panel2.setDividerLocation(300);
	
		JPanel panel01 = new JPanel();
	
		panel01.setLayout(null);
		frame_panel.setTopComponent(panel01);
	
		JLabel panel01_jLabel1 = new javax.swing.JLabel();
		panel01_jLabel1.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel01_jLabel1.setText("Where is the source image:");
		panel01_jLabel1.setBounds(10, 20, 200, 25);
		panel01.add(panel01_jLabel1);
	
		JTextField panel01_jLabel1_text = new JTextField();
		panel01_jLabel1_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_jLabel1_text.setBounds(215, 20, 250, 25);
		panel01.add(panel01_jLabel1_text);
	
		JButton panel01_choose_img_button = new JButton("Browse...");
		panel01_choose_img_button.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel01_choose_img_button.setBounds(480, 20, 110, 25);
		panel01.add(panel01_choose_img_button);
	
		panel01_choose_img_button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				JFileChooser chooser = new JFileChooser();
				chooser.setDialogTitle("Choose a source image");
				chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
				int result = chooser.showOpenDialog(null);
				if (result == JFileChooser.APPROVE_OPTION) {
					panel01_jLabel1_text.setText(chooser.getSelectedFile().getAbsolutePath());
					Prefs.set("Auto_LF_Deconvolution.imageName", chooser.getSelectedFile().getName());
					Prefs.set("Auto_LF_Deconvolution.imagePath", chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
		JCheckBox panel01_isShowsource_img = new JCheckBox("Show source image");
		panel01_isShowsource_img.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_isShowsource_img.setBounds(594, 20, 165, 25);
		panel01.add(panel01_isShowsource_img);
	
		panel01_isShowsource_img.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (panel01_isShowsource_img.isSelected()) {
					ImagePlus image = IJ.openImage(panel01_jLabel1_text.getText());
					image.show();
				} else {
					String src_imageName = Prefs.get("Auto_LF_Deconvolution.imageName", null);
					ImagePlus image = WindowManager.getImage(src_imageName);
					image.close();
				}
			}
		});
	
		JLabel panel01_jLabel2 = new javax.swing.JLabel();
		panel01_jLabel2.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel01_jLabel2.setText("Where to save the result:");
		panel01_jLabel2.setBounds(10, 50, 200, 25);
		panel01.add(panel01_jLabel2);
	
		JTextField panel01_jLabel2_text = new JTextField();
		panel01_jLabel2_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_jLabel2_text.setBounds(215, 50, 250, 25);
		panel01.add(panel01_jLabel2_text);
	
		JButton panel01_choose_save_button = new JButton("Browse...");
		panel01_choose_save_button.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_choose_save_button.setBounds(480, 50, 110, 25);
		panel01.add(panel01_choose_save_button);
		panel01_choose_save_button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				JFileChooser chooser = new JFileChooser();
				chooser.setDialogTitle("Choose a folder");
				chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
				int result = chooser.showOpenDialog(null);
				if (result == JFileChooser.APPROVE_OPTION) {
					panel01_jLabel2_text.setText(chooser.getSelectedFile().getAbsolutePath());
					Prefs.set("Auto_LF_Deconvolution.savePath", chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
	
		JCheckBox panel01_isShowper_result = new JCheckBox("Show iteration result");
		panel01_isShowper_result.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_isShowper_result.setBounds(594, 50, 175, 25);
		panel01.add(panel01_isShowper_result);
		panel01_isShowper_result.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (panel01_isShowper_result.isSelected()) {
					Prefs.set("Auto_LF_Deconvolution.is_Showper_result", true);
				} else {
					Prefs.set("Auto_LF_Deconvolution.is_Showper_result", false);
				}
			}
		});
	
		JLabel panel01_jLabel3 = new JLabel();
		panel01_jLabel3.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_jLabel3.setText("OutputName:");
		panel01_jLabel3.setBounds(10, 110, 100, 25);
		panel01.add(panel01_jLabel3);
	
		JTextField panel01_jLabel3_text = new JTextField();
		panel01_jLabel3_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_jLabel3_text.setBounds(115, 110, 150, 25);
		panel01.add(panel01_jLabel3_text);
	
		JCheckBox panel01_check1 = new JCheckBox("Save per result");
		panel01_check1.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_check1.setBounds(268, 110, 150, 25);
		panel01.add(panel01_check1);
		panel01_check1.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (panel01_check1.isSelected()) {
					Prefs.set("Auto_LF_Deconvolution.storperiter", true);
				} else {
					Prefs.set("Auto_LF_Deconvolution.storperiter", false);
				}
			}
		});
	
		JCheckBox panel01_check2 = new JCheckBox("Use psf from .mat");
		panel01_check2.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_check2.setBounds(424, 110, 160, 25);
		panel01.add(panel01_check2);
	
		JTextField panel01_jLabel4_text = new JTextField();
		panel01_jLabel4_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_jLabel4_text.setBounds(590, 110, 150, 25);
		panel01.add(panel01_jLabel4_text);
	
		JButton panel01_choose_psf_button = new JButton("Browse...");
		panel01_choose_psf_button.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_choose_psf_button.setBounds(745, 110, 110, 25);
		panel01_choose_psf_button.setEnabled(false);
		panel01.add(panel01_choose_psf_button);
		panel01_choose_psf_button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				JFileChooser chooser = new JFileChooser();
				chooser.setDialogTitle("Choose a psf from matlab");
				chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
				int result = chooser.showOpenDialog(null);
				if (result == JFileChooser.APPROVE_OPTION) {
					panel01_jLabel4_text.setText(chooser.getSelectedFile().getAbsolutePath());
					Prefs.set("Auto_LF_Deconvolution.psfPath", chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
		panel01_check2.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (panel01_check2.isSelected()) {
					panel01_choose_psf_button.setEnabled(true);
					Prefs.set("Auto_LF_Deconvolution.usepsf_from_matlab", true);
				} else {
					panel01_choose_psf_button.setEnabled(false);
					Prefs.set("Auto_LF_Deconvolution.usepsf_from_matlab", false);
				}
			}
		});
	
		JLabel panel01_jLabel4 = new JLabel();
		panel01_jLabel4.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel01_jLabel4.setText("GPUs");
		panel01_jLabel4.setBounds(800, 20, 50, 25);
		panel01.add(panel01_jLabel4);
		JTextField panel01_jLabel5_text = new JTextField();
		panel01_jLabel5_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_jLabel5_text.setBounds(800, 50, 40, 25);
		panel01.add(panel01_jLabel5_text);
	
		JLabel panel01_jLabel5 = new javax.swing.JLabel();
		panel01_jLabel5.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel01_jLabel5.setText("Where is the psf file:");
		panel01_jLabel5.setBounds(10, 80, 200, 25);
		panel01.add(panel01_jLabel5);
	
		JTextField panel01_jLabel6_text = new JTextField();
		panel01_jLabel6_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_jLabel6_text.setBounds(215, 80, 250, 25);
		panel01.add(panel01_jLabel6_text);
	
		JButton panel01_choose_psffile_button = new JButton("Browse...");
		panel01_choose_psffile_button.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_choose_psffile_button.setBounds(480, 80, 110, 25);
		panel01.add(panel01_choose_psffile_button);
		panel01_choose_psffile_button.setEnabled(false);
		panel01_choose_psffile_button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				JFileChooser chooser = new JFileChooser();
				chooser.setDialogTitle("Choose a psf file");
				chooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
				int result = chooser.showOpenDialog(null);
				if (result == JFileChooser.APPROVE_OPTION) {
					panel01_jLabel6_text.setText(chooser.getSelectedFile().getAbsolutePath());
					Prefs.set("Auto_LF_Deconvolution.PSFfilePath", chooser.getSelectedFile().getAbsolutePath());
				}
			}
		});
	
		JCheckBox panel01_isUse_psffile = new JCheckBox("Use psf file");
		panel01_isUse_psffile.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel01_isUse_psffile.setBounds(594, 80, 175, 25);
		panel01.add(panel01_isUse_psffile);
		panel01_isUse_psffile.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				if (panel01_isUse_psffile.isSelected()) {
					panel01_choose_psffile_button.setEnabled(true);
					Prefs.set("Auto_LF_Deconvolution.isUse_psffile", true);
				} else {
					panel01_choose_psffile_button.setEnabled(false);
					Prefs.set("Auto_LF_Deconvolution.isUse_psffile", false);
				}
			}
		});
	
		JPanel panel02 = new JPanel();
		panel02.setFont(new java.awt.Font(fonts, 0, 20));
		panel02.setBorder(BorderFactory.createTitledBorder(null, "Optical Parameter", TitledBorder.CENTER,
				TitledBorder.TOP, new java.awt.Font(fonts, 0, fontsize), Color.blue));
		panel02.setLayout(null);
		frame_panel2.setTopComponent(panel02);
		// frame_pane.setBottomComponent(panel02);
	
		JLabel panel02_jLabel1 = new JLabel();
		panel02_jLabel1.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel1.setText("Numerical Aperture(NA):");
		panel02_jLabel1.setBounds(20, 20, 220, 25);
		panel02.add(panel02_jLabel1);
	
		JTextField panel02_jLabel1_text = new JTextField();
		panel02_jLabel1_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel1_text.setBounds(260, 20, 100, 25);
		panel02.add(panel02_jLabel1_text);
	
		JLabel panel02_jLabel2 = new JLabel();
		panel02_jLabel2.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel2.setText("Objective Magnification(M):");
		panel02_jLabel2.setBounds(20, 60, 220, 25);
		panel02.add(panel02_jLabel2);
	
		JTextField panel02_jLabel2_text = new JTextField();
		panel02_jLabel2_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel2_text.setBounds(260, 60, 100, 25);
		panel02.add(panel02_jLabel2_text);
	
		JLabel panel02_jLabel3 = new JLabel();
		panel02_jLabel3.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel3.setText("Microlens Pitch Size(um):");
		panel02_jLabel3.setBounds(20, 100, 220, 25);
		panel02.add(panel02_jLabel3);
	
		JTextField panel02_jLabel3_text = new JTextField();
		panel02_jLabel3_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel3_text.setBounds(260, 100, 100, 25);
		panel02.add(panel02_jLabel3_text);
	
		JLabel panel02_jLabel4 = new JLabel();
		panel02_jLabel4.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel4.setText("Microlens Focal Length(um):");
		panel02_jLabel4.setBounds(20, 140, 220, 25);
		panel02.add(panel02_jLabel4);
	
		JTextField panel02_jLabel4_text = new JTextField();
		panel02_jLabel4_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel4_text.setBounds(260, 140, 100, 25);
		panel02.add(panel02_jLabel4_text);
	
		JLabel panel02_jLabel5 = new JLabel();
		panel02_jLabel5.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel5.setText("Wavelength(um):");
		panel02_jLabel5.setBounds(20, 180, 220, 25);
		panel02.add(panel02_jLabel5);
	
		JTextField panel02_jLabel5_text = new JTextField();
		panel02_jLabel5_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel5_text.setBounds(260, 180, 100, 25);
		panel02.add(panel02_jLabel5_text);
	
		JLabel panel02_jLabel6 = new JLabel();
		panel02_jLabel6.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel6.setText("Refractive Index(n):");
		panel02_jLabel6.setBounds(20, 220, 220, 25);
		panel02.add(panel02_jLabel6);
	
		JTextField panel02_jLabel6_text = new JTextField();
		panel02_jLabel6_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel6_text.setBounds(260, 220, 100, 25);
		panel02.add(panel02_jLabel6_text);
	
		JLabel panel02_jLabel7 = new JLabel();
		panel02_jLabel7.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel7.setText("Sample times each pixel(OSR):");
		panel02_jLabel7.setBounds(450, 20, 300, 25);
		panel02.add(panel02_jLabel7);
	
		JTextField panel02_jLabel7_text = new JTextField();
		panel02_jLabel7_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel7_text.setBounds(740, 20, 100, 25);
		panel02.add(panel02_jLabel7_text);
	
		JLabel panel02_jLabel8 = new JLabel();
		panel02_jLabel8.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel8.setText("Pixel numbers each microlens(Nnum):");
		panel02_jLabel8.setBounds(450, 60, 300, 25);
		panel02.add(panel02_jLabel8);
	
		JTextField panel02_jLabel8_text = new JTextField();
		panel02_jLabel8_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel8_text.setBounds(740, 60, 100, 25);
		panel02.add(panel02_jLabel8_text);
	
		JLabel panel02_jLabel9 = new JLabel();
		panel02_jLabel9.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel9.setText("z-spacing(um):");
		panel02_jLabel9.setBounds(450, 100, 300, 25);
		panel02.add(panel02_jLabel9);
	
		JTextField panel02_jLabel9_text = new JTextField();
		panel02_jLabel9_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel9_text.setBounds(740, 100, 100, 25);
		panel02.add(panel02_jLabel9_text);
	
		JLabel panel02_jLabel10 = new JLabel();
		panel02_jLabel10.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel10.setText("z-min(um):");
		panel02_jLabel10.setBounds(450, 140, 300, 25);
		panel02.add(panel02_jLabel10);
	
		JTextField panel02_jLabel10_text = new JTextField();
		panel02_jLabel10_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel10_text.setBounds(740, 140, 100, 25);
		panel02.add(panel02_jLabel10_text);
	
		JLabel panel02_jLabel11 = new JLabel();
		panel02_jLabel11.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel11.setText("z-max(um):");
		panel02_jLabel11.setBounds(450, 180, 300, 25);
		panel02.add(panel02_jLabel11);
	
		JTextField panel02_jLabel11_text = new JTextField();
		panel02_jLabel11_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel11_text.setBounds(740, 180, 100, 25);
		panel02.add(panel02_jLabel11_text);
	
		JLabel panel02_jLabel12 = new JLabel();
		panel02_jLabel12.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel12.setText("Maximum number of iterations:");
		panel02_jLabel12.setBounds(450, 220, 300, 25);
		panel02.add(panel02_jLabel12);
	
		JTextField panel02_jLabel12_text = new JTextField();
		panel02_jLabel12_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel12_text.setBounds(740, 220, 100, 25);
		panel02.add(panel02_jLabel12_text);
	
		JLabel panel02_jLabel13 = new JLabel();
		panel02_jLabel13.setFont(new java.awt.Font(fonts, 0, fontsize)); // NOI18N
		panel02_jLabel13.setText("Psfsupport diameter:");
		panel02_jLabel13.setBounds(20, 260, 220, 25);
		panel02.add(panel02_jLabel13);
	
		JTextField panel02_jLabel13_text = new JTextField();
		panel02_jLabel13_text.setFont(new java.awt.Font(fonts, 0, fontsize));
		panel02_jLabel13_text.setBounds(260, 260, 100, 25);
		panel02.add(panel02_jLabel13_text);
	
		JPanel panel03 = new JPanel();
		frame_panel2.setBottomComponent(panel03);
	
		if (os_name.equals("Linux")) {
			// output = new JTextArea(18,117);
			output = new JTextArea(17, 60);
		} else {
			output = new JTextArea(11, 60);
		}
		// DefaultCaret caret = (DefaultCaret)output.getCaret();
		// caret.setUpdatePolicy(DefaultCaret.ALWAYS_UPDATE);
		output.setFont(new java.awt.Font(fonts, 0, fontsize));
		// output.setLineWrap(true);
		output.setEditable(false);
		output.setLineWrap(true);
		output.setWrapStyleWord(true);
	
		JScrollPane output_information = new JScrollPane(output);
		panel03.add(output_information);
		output_information.setBorder(BorderFactory.createTitledBorder(null, "Output Information", TitledBorder.CENTER,
				TitledBorder.TOP, new java.awt.Font(fonts, 0, fontsize), Color.blue));
	
		JButton isOK_Button = new JButton("Start");
		isOK_Button.setForeground(Color.RED);
		isOK_Button.setFont(new java.awt.Font(fonts, 0, 28));
		isOK_Button.setBounds(450, 260, 110, 35);
		panel02.add(isOK_Button);
		isOK_Button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				String outputName = panel01_jLabel3_text.getText();
				double dna = Double.parseDouble(panel02_jLabel1_text.getText());
				double dM = Double.parseDouble(panel02_jLabel2_text.getText());
				double dMLPitch = Double.parseDouble(panel02_jLabel3_text.getText());
				int nfml = Integer.parseInt(panel02_jLabel4_text.getText());
				int nlambda = Integer.parseInt(panel02_jLabel5_text.getText());
				double dn = Double.parseDouble(panel02_jLabel6_text.getText());
				int nOSR = Integer.parseInt(panel02_jLabel7_text.getText());
				int nNnum = Integer.parseInt(panel02_jLabel8_text.getText());
				double dzmin = Double.parseDouble(panel02_jLabel10_text.getText());
				double dzmax = Double.parseDouble(panel02_jLabel11_text.getText());
				double dzspacing = Double.parseDouble(panel02_jLabel9_text.getText());
				int nIter = Integer.parseInt(panel02_jLabel12_text.getText());
				double npfsupportdiameter = Double.parseDouble(panel02_jLabel13_text.getText());
				int ngpu_num = Integer.parseInt(panel01_jLabel5_text.getText());
				String savepath = panel01_jLabel2_text.getText();
				String imagepath = panel01_jLabel1_text.getText();
				String psfmat_path = panel01_jLabel4_text.getText();
				String psf_filepath = panel01_jLabel6_text.getText();
				boolean is_Showper_result = panel01_isShowper_result.isSelected();
				boolean storperiter = panel01_check1.isSelected();
				boolean isUse_mat = panel01_check2.isSelected();
				boolean isUse_psffile = panel01_isUse_psffile.isSelected();
	
				Prefs.set("Auto_LF_Deconvolution.isUse_psffile", isUse_psffile);
				Prefs.set("Auto_LF_Deconvolution.PSFfilePath", psf_filepath);
				Prefs.set("Auto_LF_Deconvolution.storperiter", storperiter);
				Prefs.set("Auto_LF_Deconvolution.is_Showper_result", is_Showper_result);
				Prefs.set("Auto_LF_Deconvolution.psfPath", psfmat_path);
				Prefs.set("Auto_LF_Deconvolution.imagePath", imagepath);
				Prefs.set("Auto_LF_Deconvolution.savePath", savepath);
				Prefs.set("Auto_LF_Deconvolution.outputName", outputName);
				Prefs.set("Auto_LF_Deconvolution.nIter", nIter);
				Prefs.set("Auto_LF_Deconvolution.npsfsupportdiameter", npfsupportdiameter);
				Prefs.set("Auto_LF_Deconvolution.ngpu_num", ngpu_num);
				Prefs.set("Auto_LF_Deconvolution.dna", dna);
				Prefs.set("Auto_LF_Deconvolution.dM", dM);
				Prefs.set("Auto_LF_Deconvolution.dMLPitch", dMLPitch);
				Prefs.set("Auto_LF_Deconvolution.nNnum", nNnum);
				Prefs.set("Auto_LF_Deconvolution.nOSR", nOSR);
				Prefs.set("Auto_LF_Deconvolution.nfml", nfml);
				Prefs.set("Auto_LF_Deconvolution.nlambda", nlambda);
				Prefs.set("Auto_LF_Deconvolution.dn", dn);
				Prefs.set("Auto_LF_Deconvolution.dzmax", dzmax);
				Prefs.set("Auto_LF_Deconvolution.dzmin", dzmin);
				Prefs.set("Auto_LF_Deconvolution.dzspacing", dzspacing);
				Prefs.set("Auto_LF_Deconvolution.usepsf_from_matlab", isUse_mat);
				Prefs.set("Auto_LF_Deconvolution.run", true);
			}
		});
	
		JButton isStop_Button = new JButton("Stop");
		isStop_Button.setForeground(Color.RED);
		isStop_Button.setFont(new java.awt.Font(fonts, 0, 28));
		isStop_Button.setBounds(650, 260, 110, 35);
		panel02.add(isStop_Button);
		isStop_Button.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				System.exit(0);
			}
		});
	
		String outputName = Prefs.get("Auto_LF_Deconvolution.outputName", "Iter");
		panel01_jLabel3_text.setText(outputName);
		int ngpu_num = (int) Prefs.get("Auto_LF_Deconvolution.ngpu_num", 2);
		panel01_jLabel5_text.setText(String.valueOf(ngpu_num));
		double dna = Prefs.get("Auto_LF_Deconvolution.dna", 0.8);
		panel02_jLabel1_text.setText(String.valueOf(dna));
		double dM = Prefs.get("Auto_LF_Deconvolution.dM", 40);
		panel02_jLabel2_text.setText(String.valueOf(dM));
		double dMLPitch = Prefs.get("Auto_LF_Deconvolution.dMLPitch", 150);
		panel02_jLabel3_text.setText(String.valueOf(dMLPitch));
		int nfml = (int) Prefs.get("Auto_LF_Deconvolution.nfml", 3000);
		panel02_jLabel4_text.setText(String.valueOf(nfml));
		int nlambda = (int) Prefs.get("Auto_LF_Deconvolution.nlambda", 520);
		panel02_jLabel5_text.setText(String.valueOf(nlambda));
		double dn = Prefs.get("Auto_LF_Deconvolution.dn", 1.0);
		panel02_jLabel6_text.setText(String.valueOf(dn));
		int nOSR = (int) Prefs.get("Auto_LF_Deconvolution.nOSR", 3);
		panel02_jLabel7_text.setText(String.valueOf(nOSR));
		int nNnum = (int) Prefs.get("Auto_LF_Deconvolution.nNnum", 15);
		panel02_jLabel8_text.setText(String.valueOf(nNnum));
		double dzmin = Prefs.get("Auto_LF_Deconvolution.dzmin", -26.0);
		panel02_jLabel10_text.setText(String.valueOf(dzmin));
		double dzmax = Prefs.get("Auto_LF_Deconvolution.dzmax", 0.0);
		panel02_jLabel11_text.setText(String.valueOf(dzmax));
		double dzspacing = Prefs.get("Auto_LF_Deconvolution.dzspacing", 2);
		panel02_jLabel9_text.setText(String.valueOf(dzspacing));
		int nIter = (int) Prefs.get("Auto_LF_Deconvolution.nIter", 4);
		panel02_jLabel12_text.setText(String.valueOf(nIter));
		double npfsupportdiameter = Prefs.get("Auto_LF_Deconvolution.npsfsupportdiameter", 11.9);
		panel02_jLabel13_text.setText(String.valueOf(npfsupportdiameter));
		String savepath = Prefs.get("Auto_LF_Deconvolution.savePath", "/data/liutianqiang/matlab_Decon/MCF10A");
		panel01_jLabel2_text.setText(savepath);
		String imagepath = Prefs.get("Auto_LF_Deconvolution.imagePath", "/data/liutianqiang/MultiGPU_DeconJ_v5");
		panel01_jLabel1_text.setText(imagepath);
		String titleOut = Prefs.get("Auto_LF_Deconvolution.outputName", "Iter");
		panel01_jLabel3_text.setText(titleOut);
		String psfmat_path = Prefs.get("Auto_LF_Deconvolution.psfPath", null);
		panel01_jLabel4_text.setText(psfmat_path);
		boolean is_Showper_result = Prefs.get("Auto_LF_Deconvolution.is_Showper_result", false);
		panel01_isShowper_result.setSelected(is_Showper_result);
		boolean storperiter = Prefs.get("Auto_LF_Deconvolution.storperiter", false);
		panel01_check1.setSelected(storperiter);
		boolean isUse_psffile = Prefs.get("Auto_LF_Deconvolution.isUse_psffile", false);
		panel01_isUse_psffile.setSelected(isUse_psffile);
		String psf_filepath = Prefs.get("Auto_LF_Deconvolution.PSFfilePath", null);
		panel01_jLabel6_text.setText(psf_filepath);
	
		Prefs.set("Auto_LF_Deconvolution.imageName", "test.tif");
		Prefs.set("Auto_LF_Deconvolution.run", false);
		frame.setContentPane(frame_panel);
		frame.setVisible(true);
	}
	
}

class Kernelsetup {

	 static KernelLauncher cudaFilesetup(String cuName, String functionName) {
		// // Obtain the CUDA source code from the CUDA file
		Kernelsetup t = new Kernelsetup();
		InputStream cuInputStream = t.getClass().getResourceAsStream(cuName);
		String sourceCode = "";
		try {
			sourceCode = read(cuInputStream);
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				cuInputStream.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		File cuFile = null;
		try {
			cuFile = File.createTempFile("temp_JCuda_", ".ptx");
		} catch (IOException e) {
			throw new CudaException("Could not create temporary .ptx file", e);
		}
		String cuFileName = cuFile.getPath();
		FileOutputStream fos = null;
		try {
			fos = new FileOutputStream(cuFile);
			fos.write(sourceCode.getBytes());
		} catch (IOException e) {
			throw new CudaException("Could not write temporary .ptx file", e);
		} finally {
			if (fos != null) {
				try {
					fos.close();
				} catch (IOException e) {
					throw new CudaException("Could not close temporary .ptx file", e);
				}
			}
		}
		// Create the kernelLauncher that will execute the kernel
		return KernelLauncher.load(cuFileName, functionName);
	}

	private static String read(InputStream inputStream) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(inputStream));
		StringBuilder sb = new StringBuilder();
		while (true) {
			String line = br.readLine();
			if (line == null) {
				break;
			}
			sb.append(line).append("\n");
		}
		return sb.toString();
	}
}

/**
* This class is used to record psf result
*/
class Result_Pointer {

	/**
     * psf result pointer
     */		
	Pointer pointer;

	/**
     * psf result size
     */			
	int[] size;

	Result_Pointer(Pointer p,int[] s){
		pointer = p;
		size = s;
	}
}


/**
* This class is used to psf compute
*/
class Psfcompute_Thread extends Thread{

	/**
	* Z-plane starting position
	*/
	double dzmin;

	/**
	* Z-plane terminal  position
	*/	
	double dzmax;

	/**
	* Z-plane abs max  position
	*/		
	private double p3max;

	/**
	* psf max size
	*/		
	private int outarea;

	/**
	* the cuda number fot this thread
	*/			
	int device_num;

	/**
	* the result for psf
	*/		
	Result_Pointer psf_pointer_result; 

	/**
	* the result for psfht
	*/		
	Pointer Ht_pointer;

	CUcontext cuda_context;

	Psfcompute_Thread(double pmax,int out,double zmin,double zmax,int device, CUcontext context){
		p3max = pmax;
		outarea = out;
		dzmin = zmin;
		dzmax = zmax;
		device_num = device;
		cuda_context = context;
	}

	@Override
	public void run(){
		DecimalFormat df = new DecimalFormat("#.0000");
		// String savepath = Prefs.get("Auto_LF_Deconvolution.savePath", null);
		KernelLauncher[] kernelLauncher = new KernelLauncher[10];
		// JCuda.cudaSetDevice(device_num);
		// CUdevice device = new CUdevice();
		// cuDeviceGet(device, device_num);
		cuCtxSetCurrent(cuda_context);
		kernelLauncher[0] = Kernelsetup.cudaFilesetup("convolutionKernel.ptx", "convolutionKernel");
		kernelLauncher[1] = Kernelsetup.cudaFilesetup("forcomputeKernel.ptx", "forcomputeKernel");
		kernelLauncher[2] = Kernelsetup.cudaFilesetup("multiply.ptx", "multiply");
		kernelLauncher[3] = Kernelsetup.cudaFilesetup("imshift.ptx", "imshift");
		kernelLauncher[4] = Kernelsetup.cudaFilesetup("psfcomplex.ptx", "psfcomplex");
		kernelLauncher[5] = Kernelsetup.cudaFilesetup("imshiftap.ptx", "imshiftap");
		kernelLauncher[6] = Kernelsetup.cudaFilesetup("pixelbinning.ptx", "pixelbinning");
		kernelLauncher[7] = Kernelsetup.cudaFilesetup("getpsf.ptx", "getpsf");
		kernelLauncher[8] = Kernelsetup.cudaFilesetup("reductionMax.ptx","reductionMax");
		kernelLauncher[9] = Kernelsetup.cudaFilesetup("normalize.ptx","normalize");
		Auto_LF_Deconvolution.output.append(Thread.currentThread().getName()+" dzmin: "+df.format(dzmin*1e6)+" dzmax: "+df.format(dzmax*1e6)+"\n");
		Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
		psf_pointer_result =  psf_compute(dzmax, dzmin,outarea,p3max,kernelLauncher);

		int current_thread = Thread_syn.threadnum-1;
		synchronized(Thread_syn.lock){
			Thread_syn.threadnum--;
		}
		
		//Wait for the PSF of all thread to complete
		while(true){
			if(Thread_syn.threadnum == 0){
				break;
			}	
			try {
				sleep(1);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}			
		}	

		
		int thread_szie[] = psf_pointer_result.size;
		int tempH1 = thread_szie[0];
		int tempH2 = thread_szie[1];
		int tempH3 = thread_szie[2];
		int tempH4 = thread_szie[3];
		int tempH5 = thread_szie[4];

		//update the size for psf	
		synchronized(Auto_LF_Deconvolution.lock){
			Auto_LF_Deconvolution.size_H5 += tempH5;
			Auto_LF_Deconvolution.size_H1 = tempH1;
			Auto_LF_Deconvolution.size_H2 = tempH2;
			Auto_LF_Deconvolution.size_H3 = tempH3;
			Auto_LF_Deconvolution.size_H4 = tempH4;
		}
		
		int blockSize = 1024;
		int gridSize =  (int) Math.ceil((double) ((long)tempH1*tempH2*tempH3*tempH4*tempH5) / blockSize);
		kernelLauncher[8].setGridSize(gridSize, 1);
		kernelLauncher[8].setBlockSize(blockSize, 1, 1);

		Pointer temp_max = new Pointer();
		cudaMalloc(temp_max, gridSize*Sizeof.FLOAT);

		Pointer max_result = new Pointer();
		cudaMalloc(max_result, Sizeof.FLOAT);

		Pointer blocknum = new Pointer();
		cudaMalloc(blocknum, Sizeof.INT);	
		cudaMemcpy(blocknum, Pointer.to(new int[]{0}), Sizeof.INT, cudaMemcpyHostToDevice);		

		//reduction to psf max value
		kernelLauncher[8].call(psf_pointer_result.pointer,max_result,temp_max,blocknum,gridSize,
								tempH1,tempH2,tempH3,tempH4,tempH5);
		cudaFree(temp_max);
		float result[] = {0};
		cudaMemcpy(Pointer.to(result),max_result, Sizeof.FLOAT, cudaMemcpyDeviceToHost);
		cudaFree(max_result);
		Thread_syn.max_value[current_thread] = result[0];

		synchronized(Thread_syn.lock){
			Thread_syn.threadnum++;
		}
		while(true){
			if(Thread_syn.threadnum == Thread_syn.totalnum){
				break;
			}
				
			try {
				sleep(50);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}	
		}
		//get the psf max 
		float maxH = 0;
		for(int i = 0; i< Thread_syn.max_value.length; i++){
			if(maxH < Thread_syn.max_value[i]){
				maxH = Thread_syn.max_value[i];
			}
		}


		kernelLauncher[9].setGridSize(gridSize, 1);
		kernelLauncher[9].setBlockSize(blockSize, 1, 1);

		//normalize the psf
		kernelLauncher[9].call(psf_pointer_result.pointer,maxH,0,
									tempH1,tempH2,tempH3,tempH4,tempH5);

		KernelLauncher[] kernelHt = new KernelLauncher[6];
		kernelHt[0] = Kernelsetup.cudaFilesetup("getHnew.ptx", "getHnew");
		kernelHt[1] = Kernelsetup.cudaFilesetup("multiply.ptx", "multiply");
		kernelHt[2] = Kernelsetup.cudaFilesetup("add.ptx", "add");
		kernelHt[3] = Kernelsetup.cudaFilesetup("setprojection.ptx", "setprojection");
		kernelHt[4] = Kernelsetup.cudaFilesetup("backht.ptx", "backht");	
		kernelHt[5] = Kernelsetup.cudaFilesetup("normalize.ptx", "normalize");

		Auto_LF_Deconvolution.output.append(Thread.currentThread().getName()+" Compute the psf_Ht!\n");
		Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
		Ht_pointer = calculate_Ht(psf_pointer_result,kernelHt);


		kernelHt[5].setGridSize(gridSize, 1);
		kernelHt[5].setBlockSize(blockSize, 1, 1);

		//Remove inappropriate values
		kernelHt[5].call(Ht_pointer,1,1,
							tempH1,tempH2,tempH3,tempH4,tempH5);

		// ImageStack outputstack = new ImageStack(tempH1, tempH2,tempH3*tempH4*tempH5);
		// int temp_size = tempH1*tempH2*tempH3*tempH4*tempH5;
		// float[] temppsf = new float[temp_size];
		// cudaMemcpy(Pointer.to(temppsf),psf_pointer_result.pointer, (long)temp_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
		// float[][] temp_H = new float[tempH3*tempH4*tempH5][tempH1 * tempH2];
		// for(int z = 0;z<tempH5;z++){
		// 	for(int y = 0;y<tempH4;y++){
		// 		for(int x =0;x<tempH3;x++){
		// 			for(int i = 0;i<tempH1;i++){
		// 				for(int j = 0;j<tempH2;j++){
		// 					temp_H[z*tempH4*tempH3+y*tempH3+x][i*tempH2+j] 
		// 					= temppsf[z*tempH4*tempH3*tempH2*tempH1+y*tempH3*tempH2*tempH1
		// 					+x*tempH2*tempH1+i *tempH2+j];
		// 				}
		// 			}
		// 			outputstack.setPixels(temp_H[z*tempH4*tempH3+y*tempH3+x],z*tempH4*tempH3+y*tempH3+x+1);
		// 		}
		// 	}
		// }

		// ImagePlus psfoutput = new ImagePlus("psf file", outputstack);
		// psfoutput.setDimensions(tempH3, tempH4, tempH5);
		// IJ.saveAs(psfoutput, "tif",savepath + "/PSF_" + "from" + df.format(dzmin*1e6) + "to" + df.format(dzmax*1e6));

		// cudaMemcpy(Pointer.to(temppsf),Ht_pointer, (long)temp_size*Sizeof.FLOAT, cudaMemcpyDeviceToHost);
		// for(int z = 0;z<tempH5;z++){
		// 	for(int y = 0;y<tempH4;y++){
		// 		for(int x =0;x<tempH3;x++){
		// 			for(int i = 0;i<tempH1;i++){
		// 				for(int j = 0;j<tempH2;j++){
		// 					temp_H[z*tempH4*tempH3+y*tempH3+x][i*tempH2+j] 
		// 					= temppsf[z*tempH4*tempH3*tempH2*tempH1+y*tempH3*tempH2*tempH1
		// 					+x*tempH2*tempH1+i *tempH2+j];
		// 				}
		// 			}
		// 			outputstack.setPixels(temp_H[z*tempH4*tempH3+y*tempH3+x],z*tempH4*tempH3+y*tempH3+x+1);
		// 		}
		// 	}
		// }
		// psfoutput = new ImagePlus("psfHt file", outputstack);
		// psfoutput.setDimensions(tempH3, tempH4, tempH5);
		// IJ.saveAs(psfoutput, "tif",savepath + "/PSFHt_" + "from" + df.format(dzmin*1e6) + "to" + df.format(dzmax*1e6));

		// Auto_LF_Deconvolution.output.append(Thread.currentThread().getName()+"PSF save complete! " + "\n");
		// Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
	}
	private Pointer calculate_Ht(Result_Pointer device_H,KernelLauncher[] kernelLauncher) {

		int size[] = device_H.size;
		int Nnum = size[2];
		long H_size = (long)size[0]*size[1]*size[2]*size[3]*size[4]*Sizeof.FLOAT;
		int tmpsize = (int) Math.ceil((double) size[0] / Nnum);
	
		int imgsize;
		if (Math.floorMod(tmpsize, 2) == 1) {
			imgsize = (tmpsize + 2) * Nnum;
		} else {
			imgsize = (tmpsize + 3) * Nnum;
		}
		int imcenter = (int) Math.floor((double) imgsize / 2);
		int imcenterinit = imcenter - (int) Math.ceil((double) Nnum / 2);
	
		int[] xsize = new int[] { imgsize, imgsize };
		int[] mmid = new int[] { (int) Math.floor((double) size[0] / 2), (int) Math.floor((double) size[1] / 2) };
		int[] temp_exsize = new int[] { xsize[0] + mmid[0], xsize[1] + mmid[1] };
		int exsize1 = (int) Math.min(Math.pow(2.0, Math.ceil(Math.log(temp_exsize[0]) / Math.log(2))),
				128 * Math.ceil((double) temp_exsize[0] / 128));
		int exsize2 = (int) Math.min(Math.pow(2.0, Math.ceil(Math.log(temp_exsize[1]) / Math.log(2))),
				128 * Math.ceil((double) temp_exsize[1] / 128));
	
		int blockSize = 32;
		int gridSize = (int) Math.ceil((double) Math.max(exsize1, exsize2) / blockSize);
	
		kernelLauncher[0].setGridSize(gridSize, gridSize);
		kernelLauncher[0].setBlockSize(blockSize, blockSize, 1);
	
		kernelLauncher[1].setGridSize(gridSize, gridSize);
		kernelLauncher[1].setBlockSize(blockSize, blockSize, 1);
	
		int gridSize3 = (int) Math.ceil((double) Math.max(imgsize, imgsize) / blockSize);
		kernelLauncher[2].setGridSize(gridSize3, gridSize3);
		kernelLauncher[2].setBlockSize(blockSize, blockSize, 1);
	
		kernelLauncher[3].setGridSize(gridSize, gridSize);
		kernelLauncher[3].setBlockSize(blockSize, blockSize, 1);
	
		int blockSize6 = 16;
		int gridSize6 = (int) Math.ceil((double) Math.max(size[0], size[1]) / blockSize6);
		kernelLauncher[4].setGridSize(gridSize6, gridSize6, (int) Math.ceil((double) size[4] / 4));
		kernelLauncher[4].setBlockSize(blockSize6, blockSize6, 4);
	
		long fftsize = (long) exsize1 * exsize2 * 2 * Sizeof.FLOAT;
		Pointer device_projection = new Pointer();
		cudaMalloc(device_projection, fftsize);
	
		cufftHandle plan = new cufftHandle();
		JCufft.cufftPlan2d(plan, exsize1, exsize2, cufftType.CUFFT_C2C);
	
		Pointer device_Hnew = new Pointer();
		cudaMalloc(device_Hnew, fftsize);
	
		Pointer device_dst = new Pointer();
		cudaMalloc(device_dst, fftsize);
	
		long result_size = (long) imgsize * imgsize * size[4] * Sizeof.FLOAT;
		Pointer device_result = new Pointer();
		cudaMalloc(device_result, result_size);
	
		Pointer device_Ht = new Pointer();
		cudaMalloc(device_Ht, H_size);
	
		Pointer device_tempresult = new Pointer();
		cudaMalloc(device_tempresult, imgsize*imgsize*size[4]*Sizeof.FLOAT);

		for (int aa = 0; aa < Nnum; aa++) {
			for (int bb = 0; bb < Nnum; bb++) {
				kernelLauncher[3].call(device_projection, imcenterinit, exsize1, exsize2, aa, bb, imgsize, imgsize);
				JCufft.cufftExecC2C(plan, device_projection, device_projection, JCufft.CUFFT_FORWARD);
				for (int cc = 0; cc < size[4]; cc++) {
					for (int xx = 0; xx < size[2]; xx++) {
						for (int yy = 0; yy < size[3]; yy++) {
	
							kernelLauncher[0].call(device_H.pointer, device_Hnew, xx, yy, cc, mmid[0], mmid[1], exsize1, exsize2,
									size[0], size[1], size[2], size[3], size[4], 1);
							JCufft.cufftExecC2C(plan, device_Hnew, device_Hnew, JCufft.CUFFT_FORWARD);
							kernelLauncher[1].call(device_projection, device_Hnew, device_dst, exsize1, exsize2);
							JCufft.cufftExecC2C(plan, device_dst, device_dst, JCufft.CUFFT_INVERSE);
							kernelLauncher[2].call(device_dst, device_result,device_tempresult,xx, yy, cc, size[2], exsize1, exsize2,
									imgsize, imgsize, 1);
						}
					}
				}
				kernelLauncher[4].call(device_Ht, device_result, size[0], size[1], size[2], size[3], bb, aa, imcenter,
						imgsize, imgsize, (int) Math.ceil((float) Nnum / 2 - aa - 1),
						(int) Math.ceil((float) Nnum / 2 - bb - 1), size[4]);
	
			}
		}
		// cudaMemcpy(Pointer.to(Ht), device_Ht, H_size, cudaMemcpyDeviceToHost);
		JCufft.cufftDestroy(plan);
		cudaFree(device_projection);
		cudaFree(device_tempresult);
		// cudaFree(device_H);
		cudaFree(device_dst);
		cudaFree(device_Hnew);
		cudaFree(device_result);
		// cudaFree(device_Ht);
	
		return device_Ht;
	}
	
	private Result_Pointer psf_compute(double dzmax, double dzmin,int outarea,double p3max,KernelLauncher[] kernelLauncher) {
	
		int size[] = new int[5];
		double n = Prefs.get("Auto_LF_Deconvolution.dn", 1.0);
		int Nnum = (int) Prefs.get("Auto_LF_Deconvolution.nNnum", 15);
		int OSR = (int) Prefs.get("Auto_LF_Deconvolution.nOSR", 3);
		double M = Prefs.get("Auto_LF_Deconvolution.dM", 40);
		double NA = Prefs.get("Auto_LF_Deconvolution.dna", 0.95);
		double MLPitch = Prefs.get("Auto_LF_Deconvolution.dMLPitch", 150) * 1e-6;
		double fml =  Prefs.get("Auto_LF_Deconvolution.nfml", 3000) * 1e-6;
		double lambda =  Prefs.get("Auto_LF_Deconvolution.nlambda", 520) * 1e-9;
		double zmax = dzmax;
		double zmin = dzmin;
		double zspacing =  Prefs.get("Auto_LF_Deconvolution.dzspacing", 2.0)* 1e-6;
	
		double k = 2 * Math.PI * n / lambda; // the number of wave
		double alpha = Math.asin(NA / n);
		double d = fml; // optical distance between the microlens and the sensor
		double ftl = 200e-3;
		double fobj = ftl / M;
	
		double pixelPitch = MLPitch / Nnum;
		// double p3max = Math.max(Math.abs(zmax), Math.abs(zmin));
	
		int imgsize_ref = (int)Math.ceil((double) (outarea + 1) / (OSR * Nnum));
		int img_halfwidth = Math.max(Nnum * (imgsize_ref + 1), 2 * Nnum);
		double[] x1space = new double[img_halfwidth * OSR * 2 + 1];
		double[] x2space = new double[img_halfwidth * OSR * 2 + 1];
		for (int i = 0; i < x1space.length; i++) {
			x1space[i] = -1 * pixelPitch * img_halfwidth + i * pixelPitch / OSR;
			x2space[i] = -1 * pixelPitch * img_halfwidth + i * pixelPitch / OSR;
		}
		int x1length = x1space.length;
		int x2length = x2space.length;
	
		double[] x1MLspace = new double[Nnum * OSR];
		double[] x2MLspace = new double[Nnum * OSR];
		for (int i = 0; i < x1MLspace.length; i++) {
			x1MLspace[i] = -1 * (pixelPitch * (Nnum * OSR - 1) / OSR / 2) + i * pixelPitch / OSR;
			x2MLspace[i] = -1 * (pixelPitch * (Nnum * OSR - 1) / OSR / 2) + i * pixelPitch / OSR;
		}
		int x1MLlength = x1MLspace.length;
		int x2MLlength = x2MLspace.length;
	
		int[] x1centerAll = new int[(int) Math.ceil(x1length / x1MLlength) + 1];
		int[] x2centerAll = new int[(int) Math.ceil(x2length / x2MLlength) + 1];
		for (int i = 0; i < x1centerAll.length; i++) {
			x1centerAll[i] = x1MLlength * i;
			x2centerAll[i] = x2MLlength * i;
		}
	
		double[] patternML_real = new double[x1MLlength * x2MLlength];
		double[] patternML_Img = new double[x1MLlength * x2MLlength];
		for (int i = 0; i < x1MLlength; i++) {
			for (int j = 0; j < x2MLlength; j++) {
				double x1 = x1MLspace[j];
				double x2 = x2MLspace[i];
	
				double xL2norm = Math.pow(x1, 2) + Math.pow(x2, 2);
				patternML_real[i * x2MLlength + j] = Math.cos(-1 * k * xL2norm / (2 * fml));
				patternML_Img[i * x2MLlength + j] = Math.sin(-1 * k * xL2norm / (2 * fml));
			}
		}
		double[] MLcenters = new double[x1length * x2length];
		for (int i = 0; i < x1centerAll.length; i++) {
			for (int j = 0; j < x2centerAll.length; j++) {
				MLcenters[x1centerAll[i] * x1length + x2centerAll[j]] = 1;
			}
		}
	
		x1MLspace = null;
		x2MLspace = null;
		x1centerAll = null;
		x2centerAll = null;
	
		Auto_LF_Deconvolution.output.append(Thread.currentThread().getName()+" PSF_Compute_1 complete!" + "\n");
		Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
		double[] MLARRY_real = cuda_execute_conv2(MLcenters, patternML_real, x1length, x2length, x1MLlength,
				x2MLlength,kernelLauncher[0]);
		double[] MLARRY_img = cuda_execute_conv2(MLcenters, patternML_Img, x1length, x2length, x1MLlength, x2MLlength,kernelLauncher[0]);
	
		float[] MLARRY_stack = new float[x1length * x2length * 2];
		for (int i = 0; i < MLARRY_real.length; i++) {
			MLARRY_stack[2 * i] = (float) MLARRY_real[i];
			MLARRY_stack[2 * i + 1] = (float) MLARRY_img[i];
		}
		patternML_real = null;
		patternML_Img = null;
		MLcenters = null;
	
		int numpts = (int)((zmax - zmin)/zspacing +0.5)+1;
	
		double[] p3All = new double[numpts];
		for (int i = 0; i < p3All.length; i++) {
			p3All[i] = zmin + i * zspacing;
		}
	
		int centerPT = (x1length - 1) / 2;
		int halfwidth = Nnum * imgsize_ref * OSR;
	
		float[] psfwave_stack = new float[x1length * x2length * numpts * 2];
		double[] temp_stack_real = new double[x1length * x2length];
		double[] temp_stack_img = new double[x1length * x2length];
	
		long x1_size = (long) x1length * Sizeof.DOUBLE;
		Pointer x1_pointer = new Pointer();
		cudaMalloc(x1_pointer, x1_size);
		cudaMemcpy(x1_pointer, Pointer.to(x1space), x1_size, cudaMemcpyHostToDevice);
	
		long x2_size = (long) x2length * Sizeof.DOUBLE;
		Pointer x2_pointer = new Pointer();
		cudaMalloc(x2_pointer, x2_size);
		cudaMemcpy(x2_pointer, Pointer.to(x2space), x2_size, cudaMemcpyHostToDevice);
	
		Pointer real_pointer = null;
		Pointer img_pointer = null;
		for (int eachpt = 0; eachpt < numpts; eachpt++) {
			
			double p3 = p3All[eachpt];	

			double t = (Math.abs(p3) / p3max);
			double h = (long)(t*1e5)*1e-5*(double)(imgsize_ref);
			double p = Math.ceil(h);
			int imagsize_ref_il = (int) p;

			// double t = (Math.abs(p3) / p3max);
			// double h = t*imgsize_ref;
	
			// double p = Math.ceil(h);
			
			// int imagsize_ref_il = (int) Math.ceil(imgsize_ref* (Math.abs(p3) / p3max));
			// System.out.println(imagsize_ref_il);
			int halfwidth_il = Math.max(Nnum * imagsize_ref_il * OSR, 2 * Nnum * OSR);
			int min_area = Math.max(centerPT - halfwidth_il, 0);
	
			real_pointer = new Pointer();
			cudaMalloc(real_pointer, (long) x1length * x2length * Sizeof.DOUBLE);
			img_pointer = new Pointer();
			cudaMalloc(img_pointer, (long) x1length * x2length * Sizeof.DOUBLE);
	
	
	
			int blockSize = 32;
			int gridSize = (int) Math.ceil((double) (centerPT - min_area) / blockSize);
			kernelLauncher[1].setGridSize(gridSize, gridSize);
			kernelLauncher[1].setBlockSize(blockSize, blockSize, 1);
			kernelLauncher[1].call(x1_pointer, x2_pointer, real_pointer, img_pointer, centerPT, alpha, M, k, x1length,
					x2length, p3, min_area, fobj, lambda);
	
			// Copy the data from the device back to the host and clean up
			cudaMemcpy(Pointer.to(temp_stack_real), real_pointer, (long) x1length * x2length * Sizeof.DOUBLE,
					cudaMemcpyDeviceToHost);
			cudaMemcpy(Pointer.to(temp_stack_img), img_pointer, (long) x1length * x2length * Sizeof.DOUBLE,
					cudaMemcpyDeviceToHost);
			
	
			for (int i = 0; i < x1length; i++) {
				for (int j = 0; j < x2length; j++) {
					psfwave_stack[eachpt * x1length * x2length * 2
							+ 2 * (i * x2length + j)] = (float) temp_stack_real[i * x2length + j];
					psfwave_stack[eachpt * x1length * x2length * 2 + 2 * (i * x2length + j)
							+ 1] = (float) temp_stack_img[i * x2length + j];
				}
			}
			
	
	
			cudaFree(real_pointer);
			cudaFree(img_pointer);
		}
		cudaFree(x1_pointer);
		cudaFree(x2_pointer);
		x1space = null;
		x2space = null;
		temp_stack_img = null;
		temp_stack_real = null;
	
		double[] x1objspace = new double[Nnum];
		double[] x2objspace = new double[Nnum];
		for (int i = 0; i < Nnum; i++) {
			x1objspace[i] = -1 * Math.floor((double) Nnum / 2) * pixelPitch / M + i * pixelPitch / M;
			x2objspace[i] = -1 * Math.floor((double) Nnum / 2) * pixelPitch / M + i * pixelPitch / M;
		}
		double[] x3objspace = new double[(int)((dzmax - dzmin) / zspacing+0.5) + 1];
		for (int i = 0; i < x3objspace.length; i++) {
			x3objspace[i] = zmin + zspacing * i;
		}
		int xref = (x1objspace.length - 1) / 2;
		int yref = (x2objspace.length - 1) / 2;
		int[] cp = new int[2 * (halfwidth / OSR) + 1];
		for (int i = 0; i < cp.length; i++) {
			cp[i] = centerPT / OSR + 1 - halfwidth / OSR + i;
		}
		double dx0 = pixelPitch / OSR;
		double du = 1 / (dx0 * x1length);
		double dv = 1 / (dx0 * x2length);
		double[] u = new double[x1length];
		double[] v = new double[x2length];
		for (int i = 0; i < x1length; i++) {
			if (i < centerPT) {
				u[i] = i * du;
				v[i] = i * dv;
			} else {
				u[i] = (-1 * centerPT + i - centerPT - 1) * du;
				v[i] = (-1 * centerPT + i - centerPT - 1) * dv;
			}
		}
		float[] tempH = new float[x1length * x2length * 2];
		for (int i = 0; i < x1length; i++) {
			for (int j = 0; j < x2length; j++) {
				tempH[2 * (i * x2length + j)] = (float) Math
						.cos(-2 * Math.PI * Math.PI * (u[i] * u[i] + v[j] * v[j]) * d / k);
				tempH[2 * (i * x2length + j) + 1] = (float) Math
						.sin(-2 * Math.PI * Math.PI * (u[i] * u[i] + v[j] * v[j]) * d / k);
			}
		}
		int xmin = Math.max(centerPT - halfwidth, 0);
		int xmax = Math.min(centerPT + halfwidth + 1, x1length);
		int ymin = Math.max(centerPT - halfwidth, 0);
		int ymax = Math.min(centerPT + halfwidth + 1, x2length);
	
		int x1centerinit = centerPT - (OSR - 1) / 2 + 1;
		int x2centerinit = centerPT - (OSR - 1) / 2 + 1;
	
		int x1init = (int) (x1centerinit - Math.floor((double) x1centerinit / OSR) * OSR);
		int x2init = (int) (x2centerinit - Math.floor((double) x1centerinit / OSR) * OSR);
	
		int x1shift = 0;
		int x2shift = 0;
		if (x1init < 1) {
			x1init = x1init + OSR - 1;
			x1shift = 1;
		}
		if (x2init < 1) {
			x2init = x2init + OSR - 1;
			x2shift = 1;
		}
		int f1_AP_halfwidth = centerPT - x1init;
	
		long psfH_length = (long)cp.length * cp.length * Nnum * Nnum * x3objspace.length;
		size[0] = cp.length;
		size[1] = cp.length;
		size[2] = Nnum;
		size[3] = Nnum;
		size[4] = numpts;
	
		Auto_LF_Deconvolution.output.append(Thread.currentThread().getName()+" PSF_Compute_2 complete!" + "\n");
		Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
	
		long device_psfwavesize = (long) psfwave_stack.length * Sizeof.FLOAT;
		Pointer device_psfwave = new Pointer();
		cudaMalloc(device_psfwave, device_psfwavesize);
		cudaMemcpy(device_psfwave, Pointer.to(psfwave_stack), device_psfwavesize, cudaMemcpyHostToDevice);
	
		long device_Hsize = (long) tempH.length * Sizeof.FLOAT;
		Pointer device_H = new Pointer();
		cudaMalloc(device_H, device_Hsize);
		cudaMemcpy(device_H, Pointer.to(tempH), device_Hsize, cudaMemcpyHostToDevice);
	
		long device_MLARRYsize = (long) MLARRY_stack.length * Sizeof.FLOAT;
		Pointer device_MLARR = new Pointer();
		cudaMalloc(device_MLARR, device_MLARRYsize);
		cudaMemcpy(device_MLARR, Pointer.to(MLARRY_stack), device_MLARRYsize, cudaMemcpyHostToDevice);
	
		long device_imshift_psfsize = (long) x1length * x2length * 2 * Sizeof.FLOAT;
		Pointer device_imshif_psfwave = new Pointer();
		cudaMalloc(device_imshif_psfwave, device_imshift_psfsize);
	
		long device_imshift_psfAPsize = (long) x1length * x2length * Sizeof.FLOAT;
		Pointer device_imshif_psfwaveAP = new Pointer();
		cudaMalloc(device_imshif_psfwaveAP, device_imshift_psfAPsize);
	
		long out_img_size = (long) (2 * f1_AP_halfwidth + 1) * (2 * f1_AP_halfwidth + 1) / OSR / OSR * Sizeof.FLOAT;
		Pointer device_outimg = new Pointer();
		cudaMalloc(device_outimg, out_img_size);
	
		long psf_size = (long) psfH_length * Sizeof.FLOAT;
		Pointer device_psf = new Pointer();
		cudaMalloc(device_psf, psf_size);
		// cudaMemcpy(device_psf, Pointer.to(psf_H), psf_size, cudaMemcpyHostToDevice);
	
		cufftHandle plan = new cufftHandle();
		JCufft.cufftPlan2d(plan, x1length, x1length, cufftType.CUFFT_C2C);
		int blockSize = 32;
		int gridSize = (int) Math.ceil((double) Math.max(x1length, x2length) / blockSize);
	
		kernelLauncher[2].setGridSize(gridSize, gridSize);
		kernelLauncher[2].setBlockSize(blockSize, blockSize, 1);
	
		kernelLauncher[3].setGridSize(gridSize, gridSize);
		kernelLauncher[3].setBlockSize(blockSize, blockSize, 1);
	
		kernelLauncher[4].setGridSize(gridSize, gridSize);
		kernelLauncher[4].setBlockSize(blockSize, blockSize, 1);
	
		kernelLauncher[5].setGridSize(gridSize, gridSize);
		kernelLauncher[5].setBlockSize(blockSize, blockSize, 1);
	
		int gridSize7 = (int) Math.ceil((double) (2 * f1_AP_halfwidth + 1) / OSR / blockSize);
		kernelLauncher[6].setGridSize(gridSize7, gridSize7);
		kernelLauncher[6].setBlockSize(blockSize, blockSize, 1);
	
		int gridSize8 = (int) Math.ceil((double) (size[0]) / blockSize);
		kernelLauncher[7].setGridSize(gridSize8, gridSize8);
		kernelLauncher[7].setBlockSize(blockSize, blockSize, 1);
	
		for (int z = 0; z < x3objspace.length; z++) {
			for (int x = 0; x < x1objspace.length; x++) {
				for (int y = 0; y < x2objspace.length; y++) {
	
					kernelLauncher[3].call(device_imshif_psfwave, device_psfwave, z, x1length, x2length, OSR * (x - xref),
							OSR * (y - yref), 1);
					kernelLauncher[3].call(device_imshif_psfwave, device_psfwave, z, x1length, x2length, OSR * (x - xref),
							OSR * (y - yref), 0);
	
					kernelLauncher[2].call(device_imshif_psfwave, device_MLARR, device_imshif_psfwave, x1length,
							x2length);
					JCufft.cufftExecC2C(plan, device_imshif_psfwave, device_imshif_psfwave, JCufft.CUFFT_FORWARD);
					kernelLauncher[2].call(device_imshif_psfwave, device_H, device_imshif_psfwave, x1length, x2length);
					JCufft.cufftExecC2C(plan, device_imshif_psfwave, device_imshif_psfwave, JCufft.CUFFT_INVERSE);
	
					kernelLauncher[4].call(device_imshif_psfwave, device_imshif_psfwave, k, d, x1length, x2length);
	
					kernelLauncher[5].call(device_imshif_psfwaveAP, device_imshif_psfwave, xmin, xmax, ymin, ymax,
							x1length, x2length, -1 * OSR * (x - xref), -1 * OSR * (y - yref), 1);
					kernelLauncher[5].call(device_imshif_psfwaveAP, device_imshif_psfwave, xmin, xmax, ymin, ymax,
							x1length, x2length, -1 * OSR * (x - xref), -1 * OSR * (y - yref), 0);
	
					kernelLauncher[6].call(device_imshif_psfwaveAP, device_outimg, x1init, x2init, x1length, x2length,
							(2 * f1_AP_halfwidth + 1) / OSR, (2 * f1_AP_halfwidth + 1) / OSR, OSR, 1);
					kernelLauncher[6].call(device_imshif_psfwaveAP, device_outimg, x1init, x2init, x1length, x2length,
							(2 * f1_AP_halfwidth + 1) / OSR, (2 * f1_AP_halfwidth + 1) / OSR, OSR, 0);
	
					kernelLauncher[7].call(device_outimg, device_psf, x1shift, x2shift, cp[0], size[0], size[1], size[2],
						size[3], size[4], (2 * f1_AP_halfwidth + 1) / OSR, (2 * f1_AP_halfwidth + 1) / OSR, z, y,
							x);
				}
			}
		}
		
		cudaFree(device_H);
		cudaFree(device_imshif_psfwave);
		cudaFree(device_psfwave);
		cudaFree(device_MLARR);
		// cudaFree(device_psf);
		cudaFree(device_imshif_psfwaveAP);
		cudaFree(device_outimg);
		JCufft.cufftDestroy(plan);
		u = null;
		v = null;
		tempH = null;
		cp = null;
		x1objspace = null;
		x2objspace = null;
		x3objspace = null;
	
		Auto_LF_Deconvolution.output.append(Thread.currentThread().getName()+" PSF_Compute_3 complete!" + "\n");
		Auto_LF_Deconvolution.output.setCaretPosition(Auto_LF_Deconvolution.output.getDocument().getLength());
		return new Result_Pointer(device_psf,size) ;
	
	}
	
	private double[] cuda_execute_conv2(double[] src, double[] mask, int src_w, int src_h, int mask_w, int mask_h,KernelLauncher kernelLauncher) {
	
		float[] tempsrc = new float[src_w * src_h];
		for (int i = 0; i < src_h; i++) {
			for (int j = 0; j < src_w; j++) {
				tempsrc[i * src_w + j] = (float) src[i * src_w + j];
			}
		}
		float[] tempmask = new float[mask_w * mask_h];
		for (int i = 0; i < mask_h; i++) {
			for (int j = 0; j < mask_w; j++) {
				tempmask[i * mask_w + j] = (float) mask[i * mask_w + j];
			}
		}
		float[] tempdst = new float[src_w * src_h];
		double[] dst = new double[src_w * src_h];
		int src_size = src_w * src_h * Sizeof.FLOAT;
		Pointer src_pointer = new Pointer();
		cudaMalloc(src_pointer, src_size);
		cudaMemcpy(src_pointer, Pointer.to(tempsrc), src_size, cudaMemcpyHostToDevice);
	
		long mask_size = (long) mask_w * mask_h * Sizeof.FLOAT;
		Pointer mask_pointer = new Pointer();
		cudaMalloc(mask_pointer, mask_size);
		cudaMemcpy(mask_pointer, Pointer.to(tempmask), mask_size, cudaMemcpyHostToDevice);
	
		int blockSize = 32;
		int gridSize = (int) Math.ceil((double) Math.max(src_w, src_h) / blockSize);
	
		Pointer dst_pointer = new Pointer();
		cudaMalloc(dst_pointer, src_size);
	
		kernelLauncher.setGridSize(gridSize, gridSize);
		kernelLauncher.setBlockSize(blockSize, blockSize, 1);
		kernelLauncher.call(src_pointer, mask_pointer, dst_pointer, src_w, src_h, mask_w, mask_h);
	
		// Copy the data from the device back to the host and clean up
		cudaMemcpy(Pointer.to(tempdst), dst_pointer, src_size, cudaMemcpyDeviceToHost);
		for (int i = 0; i < src_h; i++) {
			for (int j = 0; j < src_w; j++) {
				dst[i * src_w + j] = tempdst[i * src_w + j];
			}
		}
		cudaFree(src_pointer);
		cudaFree(mask_pointer);
		cudaFree(dst_pointer);
		return dst;
	}
}

