#!/bin/bash
###
 # @Author: your name
 # @Date: 2021-02-03 21:00:04
 # @LastEditTime: 2021-02-04 11:13:00
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: \LightFieldMicroscopy_ImageJPlugin\build_cuda.sh
### 
cd $(dirname $0) & echo cd:$(pwd)
nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -V
echo build add.cu to add.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/add.cu -o ./src/main/resources/add.ptx 
echo build convolutionKernel.cu to convolutionKernel.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/convolutionKernel.cu -o ./src/main/resources/convolutionKernel.ptx 
echo build forcomputeKernel.cu to forcomputeKernel.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/forcomputeKernel.cu -o ./src/main/resources/forcomputeKernel.ptx 
echo build getHnew.cu to getHnew.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/getHnew.cu -o ./src/main/resources/getHnew.ptx 
echo build getprojection.cu to getprojection.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/getprojection.cu -o ./src/main/resources/getprojection.ptx 
echo build imshift.cu to imshif.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/imshift.cu -o ./src/main/resources/imshift.ptx 
echo build multiply.cu to multiply.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/multiply.cu -o ./src/main/resources/multiply.ptx 
echo build setprojection.cu to setprojection.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/setprojection.cu -o ./src/main/resources/setprojection.ptx 
echo build psfcomplex.cu to psfcomplex.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/psfcomplex.cu -o ./src/main/resources/psfcomplex.ptx 
echo build imshiftap.cu to imshiftap.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/imshiftap.cu -o ./src/main/resources/imshiftap.ptx 
echo build getpsf.cu to getpsf.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/getpsf.cu -o ./src/main/resources/getpsf.ptx 
echo build backht.cu to backht.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/backht.cu -o ./src/main/resources/backht.ptx 
echo build pixelbinning.cu to pixelbinning.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/pixelbinning.cu -o ./src/main/resources/pixelbinning.ptx
echo build getpsf_fromGPU.cu to getpsf_fromGPU.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/getpsf_fromGPU.cu -o ./src/main/resources/getpsf_fromGPU.ptx
echo build reductionMax.cu to reductionMax.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/reductionMax.cu -o ./src/main/resources/reductionMax.ptx
echo build normalize.cu to normalize.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/normalize.cu -o ./src/main/resources/normalize.ptx
echo build forwardADD.cu to forwardADD.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/forwardADD.cu -o ./src/main/resources/forwardADD.ptx
echo build xguessmax.cu to xguessmax.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/xguessmax.cu -o ./src/main/resources/xguessmax.ptx
echo build tozero.cu to tozero.ptx & nvcc -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_60,code=compute_60 -ptx ./src/main/resources/tozero.cu -o ./src/main/resources/tozero.ptx