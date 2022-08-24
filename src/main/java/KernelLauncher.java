/*
 * JCudaUtils - Utilities for JCuda 
 * http://www.jcuda.org
 *
 * Copyright (c) 2010 Marco Hutter - http://www.jcuda.org
 * 
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

import static jcuda.driver.JCudaDriver.*;

import java.io.*;
import java.util.logging.Logger;

import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.dim3;

/**
 * This is a utility class that simplifies the setup and launching
 * of CUDA kernels using the JCuda Driver API. 
 */
public class KernelLauncher
{
    /**
     * The logger used in this class
     */
    private static final Logger logger = 
        Logger.getLogger(KernelLauncher.class.getName());
    
    /**
     * The path prefix, containing the path to the NVCC compiler.
     * Not required if the path to the NVCC is present in an
     * environment variable.
     */
    private static String compilerPath = "";
    
    /**
     * The number of the device which should be used by the
     * KernelLauncher
     */
    private static int deviceNumber = 0;
    
    /**
     * Set the path to the NVCC compiler. For example: 
     * 
     * By default, this path is empty, assuming that the compiler 
     * is in a path that is visible via an environment variable.
     * 
     * @param path The path to the NVCC compiler.
     */
    public static void setCompilerPath(String path)
    {
       if (path == null)
       {
           compilerPath = "";
       }
       compilerPath = path;
       if (!compilerPath.endsWith(File.separator))
       {
           compilerPath += File.separator;
       }
    }

    /**
     * Set the number (index) of the device which should be used
     * by the KernelLauncher
     */
    public static void setDeviceNumber(int number)
    {
    	int count[] = new int[1];
    	cuDeviceGetCount(count);
    	if (number < 0 || number >= count[0])
    	{
    		throw new CudaException(
    		    "Invalid device number: "+number+". "+
    		    "There are only "+count[0]+" devices available");
    	}
    	deviceNumber = number;
    }
    
    /**
     * Create a new KernelLauncher for the function with the given 
     * name, that is defined in the given source code. 
     */
    public static KernelLauncher compile(
        String sourceCode, String functionName, String ... nvccArguments)
    {
        File cuFile = null;
        try
        {
            cuFile = File.createTempFile("temp_JCuda_", ".cu");
        }
        catch (IOException e)
        {
            throw new CudaException("Could not create temporary .cu file", e);
        }
        String cuFileName = cuFile.getPath();
        FileOutputStream fos = null;
        try
        {
            fos = new FileOutputStream(cuFile);
            fos.write(sourceCode.getBytes());
        }
        catch (IOException e)
        {
            throw new CudaException("Could not write temporary .cu file", e);
        }
        finally
        {
            if (fos != null)
            {
                try
                {
                    fos.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                        "Could not close temporary .cu file", e);
                }
            }
        }
        return create(cuFileName, functionName, nvccArguments);
    }
    
    
    
    /**
     * Create a new KernelLauncher for the function with the given 
     * name, that is contained in the .CU CUDA source file with the
     */
    public static KernelLauncher create(
        String cuFileName, String functionName, String ... nvccArguments)
    {
        return create(cuFileName, functionName, false, nvccArguments);
    }
    
    /**
     * Create a new KernelLauncher for the function with the given 
     * name, that is contained in the .CU CUDA source file with the
     */
    public static KernelLauncher create(
        String cuFileName, String functionName, 
        boolean forceRebuild, String ... nvccArguments)
    {

        // Prepare the PTX file for the CU source file
        String ptxFileName = null;
        try
        {
            ptxFileName = 
                preparePtxFile(cuFileName, forceRebuild, nvccArguments);
        }
        catch (IOException e)
        {
            throw new CudaException(
                "Could not prepare PTX for source file '"+cuFileName+"'", e);
        }
        
        KernelLauncher kernelLauncher = new KernelLauncher();
        byte ptxData[] = loadData(ptxFileName);
        kernelLauncher.initModule(ptxData);
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }

    /**
     * Create a new KernelLauncher which may be used to execute the
     * specified function which is loaded from the PTX- or CUBIN
     */
    public static KernelLauncher load(
        String moduleFileName, String functionName)
    {
        KernelLauncher kernelLauncher = new KernelLauncher();
        byte moduleData[] = loadData(moduleFileName);
        kernelLauncher.initModule(moduleData);
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }

    /**
     * Create a new KernelLauncher which may be used to execute the
     * specified function which is loaded from the PTX- or CUBIN 
     * data that is read from the given input stream.
     */
    public static KernelLauncher load(
        InputStream moduleInputStream, String functionName)
    {
        KernelLauncher kernelLauncher = new KernelLauncher();
        byte moduleData[] = loadData(moduleInputStream);
        kernelLauncher.initModule(moduleData);
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }
    
    
    /**
     * Load the data from the file with the given name and returns 
     * it as a 0-terminated byte array
     */
    private static byte[] loadData(String fileName)
    {
        InputStream inputStream = null;
        try
        {
            inputStream= new FileInputStream(new File(fileName));
            return loadData(inputStream);
        }
        catch (FileNotFoundException e)
        {
            throw new CudaException(
                "Could not open '"+fileName+"'", e);
        }
        finally
        {
            if (inputStream != null)
            {
                try
                {
                    inputStream.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                        "Could not close '"+fileName+"'", e);
                }
            }
        }
    }
    
    /**
     * Reads the data from the given inputStream and returns it as
     * a 0-terminated byte array.
     */
    private static byte[] loadData(InputStream inputStream)
    {
        ByteArrayOutputStream baos = null;
        try
        {
            baos = new ByteArrayOutputStream();
            byte buffer[] = new byte[8192];
            while (true)
            {
                int read = inputStream.read(buffer);
                if (read == -1)
                {
                    break;
                }
                baos.write(buffer, 0, read);
            }
            baos.write('\0');
            baos.flush();
            return baos.toByteArray();
        }
        catch (IOException e)
        {
            throw new CudaException(
                "Could not load data", e);
        }
        finally
        {
            if (baos != null)
            {
                try
                {
                    baos.close();
                }
                catch (IOException e)
                {
                    throw new CudaException(
                        "Could not close output", e);
                }
            }
        }
        
    }
    

    /**
     * The context which was used to create this instance
     */
    private CUcontext context;
    
    /**
     * The module which contains the function
     */
    private CUmodule module;
    
    /**
     * The function which is executed with this KernelLauncher
     */
    private CUfunction function;
    
    /**
     * The current block size (number of threads per block)
     * which will be used for the function call.
     */
    private dim3 blockSize = new dim3(1,1,1);
    
    /**
     * The current grid size (number of blocks per grid)
     * which will be used for the function call.
     */
    private dim3 gridSize = new dim3(1,1,1);
    
    /**
     * The currently specified size of the shared memory
     * for the function call.
     */
    private int sharedMemSize = 0;
    
    /**
     * The stream that should be associated with the function call.
     */
    private CUstream stream;

    
    /**
     * Private constructor. Instantiation only via the static
     * methods.
     */
    private KernelLauncher()
    {
        initialize();        
    }
    
    /**
     * Initializes this KernelLauncher. This method will try to 
     * initialize the JCuda driver API. Then it will try to 
     * attach to the current CUDA context. If no active CUDA 
     * context exists, then it will try to create one, for
     * the device which is specified by the current 
     * deviceNumber.
     */
    private void initialize()
    {
        int result = cuInit(0);
        if (result != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(
                "Failed to initialize the driver: "+
                CUresult.stringFor(result));
        }

        // Try to obtain the current context
        context = new CUcontext();
        result = cuCtxGetCurrent(context);
        if (result != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(
                "Failed to obtain the current context: "+
                CUresult.stringFor(result));
        }
        
        // If the context is 'null', then a new context
        // has to be created.
        CUcontext nullContext = new CUcontext(); 
        if (context.equals(nullContext))
        {
            createContext();
        }
    }
    
    /**
     * Tries to create a context for device 'deviceNumber'.
 
     */
    private void createContext()
    {
        CUdevice device = new CUdevice();
        int result = cuDeviceGet(device, deviceNumber);
        if (result != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(
                "Failed to obtain a device: "+
                CUresult.stringFor(result));
        }
        
        result = cuCtxCreate(context, 0, device);
        if (result != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(
                "Failed to create a context: "+
                CUresult.stringFor(result));
        }
    }
    

    /**
     * Create a new KernelLauncher which uses the same module as
     * this KernelLauncher, but may be used to execute a different 
     * function. All parameters (grid size, block size, shared 
     * memory size and stream) of the returned KernelLauncher 
     * will be independent of 'this' one and initially contain 
     * the default values.
     */
    public KernelLauncher forFunction(String functionName)
    {
        KernelLauncher kernelLauncher = new KernelLauncher();
        kernelLauncher.module = this.module;
        kernelLauncher.initFunction(functionName);
        return kernelLauncher;
    }
    
    
    /**
     * Initialize the module for this KernelLauncher by loading
     * the PTX- or CUBIN file with the given name.
     */
    private void initModule(byte moduleData[])
    {
        module = new CUmodule();
        checkResult(cuModuleLoadDataEx(module, Pointer.to(moduleData), 
            0, new int[0], Pointer.to(new int[0])));
    }

    /**
     * Initialize this KernelLauncher for calling the function with
     * the given name, which is contained in the module of this
     * KernelLauncher
     */
    private void initFunction(String functionName)
    {
        // Obtain the function from the module
        function = new CUfunction();
        String functionErrorString =
            "Could not get function '"+functionName+"' from module. "+"\n"+
            "Name in module might be mangled. Try adding the line "+"\n"+
            "extern \"C\""+"\n"+
            "before the function you want to call, or open the " +
            "PTX/CUBIN "+"\n"+"file with a text editor to find out " +
            "the mangled function name";
        try
        {
            int result = cuModuleGetFunction(function, module, functionName);
            if (result != CUresult.CUDA_SUCCESS)
            {
                throw new CudaException(functionErrorString);
            }
        }
        catch (CudaException e)
        {
            throw new CudaException(functionErrorString, e);
        }
    }
    
    /**
     * Returns the module that was created from the PTX- or CUBIN file, and 
     * which contains the function that should be executed. This
     * module may also be used to access symbols and texture 
     * references. However, clients should not modify or unload
     * the module.
     */
    public CUmodule getModule()
    {
        return module;
    }

    /**
     * Set the grid size (number of blocks per grid) for the function 
     */
    public KernelLauncher setGridSize(int x, int y)
    {
        gridSize.x = x;
        gridSize.y = y;
        return this;
    }

    /**
     * Set the grid size (number of blocks per grid) for the function 
     */
    public KernelLauncher setGridSize(int x, int y, int z)
    {
        gridSize.x = x;
        gridSize.y = y;
        gridSize.z = z;
        return this;
    }

    /**
     * Set the block size (number of threads per block) for the function 
     */
    public KernelLauncher setBlockSize(int x, int y, int z)
    {
        blockSize.x = x;
        blockSize.y = y;
        blockSize.z = z;
        return this;
    }
    
    /**
     * Set the size of the shared memory for the function 
     */
    public KernelLauncher setSharedMemSize(int sharedMemSize)
    {
        this.sharedMemSize = sharedMemSize;
        return this;
    }

    /**
     * Set the stream for the function call.
     */
    public KernelLauncher setStream(CUstream stream)
    {
        this.stream = stream;
        return this;
    }
    
    

    /**
     * Set the given grid size and block size for this KernelLauncher.
     */
    public KernelLauncher setup(dim3 gridSize, dim3 blockSize)
    {
        return setup(gridSize, blockSize, sharedMemSize, stream);
    }
    
    /**
     * Set the given grid size and block size and shared memory size
     * for this KernelLauncher.
     */
    public KernelLauncher setup(dim3 gridSize, dim3 blockSize, 
        int sharedMemSize)
    {
        return setup(gridSize, blockSize, sharedMemSize, stream);
    }

    /**
     * Set the given grid size and block size, shared memory size
     * and stream for this KernelLauncher.
     */
    public KernelLauncher setup(dim3 gridSize, dim3 blockSize, 
        int sharedMemSize, CUstream stream)
    {
        setGridSize(gridSize.x, gridSize.y);
        setBlockSize(blockSize.x, blockSize.y, blockSize.z);
        setSharedMemSize(sharedMemSize);
        setStream(stream);
        return this;
    }
    
    /**
     * Call the function of this KernelLauncher with the current
     * grid size, block size, shared memory size and stream, and
     * with the given arguments.
     */
    public void call(Object ... args)
    {
        Pointer kernelParameters[] = new Pointer[args.length];
        
        for (int i=0; i<args.length; i++)
        {
            Object arg = args[i];
            if (arg instanceof Pointer)
            {
                Pointer argPointer = (Pointer)arg;
                Pointer pointer = Pointer.to(argPointer);
                kernelParameters[i] = pointer;
                logger.fine("argument "+i+" type is Pointer");
            }
            else if (arg instanceof Byte)
            {
                Byte value = (Byte)arg;
                Pointer pointer = Pointer.to(new byte[]{value});
                kernelParameters[i] = pointer;
                logger.fine("argument "+i+" type is Byte");
            }
            else if (arg instanceof Short)
            {
                Short value = (Short)arg;
                Pointer pointer = Pointer.to(new short[]{value});
                kernelParameters[i] = pointer;
                logger.fine("argument "+i+" type is Short");
            }
            else if (arg instanceof Integer)
            {
                Integer value = (Integer)arg;
                Pointer pointer = Pointer.to(new int[]{value});
                kernelParameters[i] = pointer;
                logger.fine("argument "+i+" type is Integer");
            }
            else if (arg instanceof Long)
            {
                Long value = (Long)arg;
                Pointer pointer = Pointer.to(new long[]{value});
                kernelParameters[i] = pointer;
                logger.fine("argument "+i+" type is Long");
            }
            else if (arg instanceof Float)
            {
                Float value = (Float)arg;
                Pointer pointer = Pointer.to(new float[]{value});
                kernelParameters[i] = pointer;
                logger.fine("argument "+i+" type is Float");
            }
            else if (arg instanceof Double)
            {
                Double value = (Double)arg;
                Pointer pointer = Pointer.to(new double[]{value});
                kernelParameters[i] = pointer;
                logger.fine("argument "+i+" type is Double");
            }
            else
            {
                throw new CudaException(
                    "Type "+arg.getClass()+" may not be passed to a function");
            }
        }
        checkResult(cuLaunchKernel(function,
            gridSize.x,  gridSize.y, gridSize.z,
            blockSize.x, blockSize.y, blockSize.z,
            sharedMemSize, stream,
            Pointer.to(kernelParameters), null
        ));
        checkResult(cuCtxSynchronize());
    }

    /**
     * If the given result is not CUresult.CUDA_SUCCESS, then this method
     * throws a CudaException with the error message for the given result.
     */
    private static void checkResult(int cuResult)
    {
        if (cuResult != CUresult.CUDA_SUCCESS)
        {
            throw new CudaException(CUresult.stringFor(cuResult));
        }
    }
    
    
    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist or is older
     * than the source file, it is compiled from the given file 
     * using NVCC. If the forceRebuild flag is 'true', then the PTX 
     * file is rebuilt even if it already exists or is newer than the
     * source file. The name of the PTX file is returned. 
     */
    private static String preparePtxFile(
        String cuFileName, boolean forceRebuild, String ... nvccArguments) 
        throws IOException
    {
        logger.config("Preparing PTX for \n"+cuFileName);

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new CudaException("Input file not found: "+cuFileName);
        }
        
        // Replace the file extension with "ptx"
        String ptxFileName = null;
        int lastIndex = cuFileName.lastIndexOf('.');
        if (lastIndex == -1)
        {
            ptxFileName = cuFileName + ".ptx";
        }
        else
        {
            ptxFileName = cuFileName.substring(0, lastIndex)+".ptx";
        }
        
        // Return if the file already exists and should not be rebuilt
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists() && !forceRebuild)
        {
            long cuLastModified = cuFile.lastModified();
            long ptxLastModified = ptxFile.lastModified();
            if (cuLastModified < ptxLastModified)
            {
                return ptxFileName;
            }
        }
        
        // Build the command line
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String defaultArguments = "";
        String optionalArguments = createArgumentsString(nvccArguments);
        String command = 
            compilerPath + "nvcc " + modelString + " " + defaultArguments + 
            " " + optionalArguments + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;

        
        // Execute the command line and wait for the output
        logger.config("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);
        String errorMessage = 
            new String(toByteArray(process.getErrorStream()));
        String outputMessage = 
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new CudaException(
                "Interrupted while waiting for nvcc output", e);
        }

        logger.config("nvcc process exitValue "+exitValue);
        if (exitValue != 0)
        {
            logger.severe("errorMessage:\n"+errorMessage);
            logger.severe("outputMessage:\n"+outputMessage);
            throw new CudaException(
                "Could not create .ptx file: "+errorMessage);
        }
        return ptxFileName;
    }
    
    /**
     * Creates a single string from the given argument strings
     */
    private static String createArgumentsString(String ... nvccArguments)
    {
        if (nvccArguments == null || nvccArguments.length == 0)
        {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (String s : nvccArguments)
        {
            sb.append(s);
            sb.append(" ");
        }
        return sb.toString();
    }
    

    /**
     * Fully reads the given InputStream and returns it as a byte array.
     */
    private static byte[] toByteArray(
        InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
    
}



