@echo off
cd /d %~dp0 & echo %~dp0
echo Build add.cu to add.ptx & ^
nvcc -ptx ./src/java/kernel/add.cu -o ./src/java/kernel/add.ptx -Xcompiler /wd4819 
echo Build convolutionKernel.cu to convolutionKernel.ptx & ^
nvcc -ptx ./src/java/kernel/convolutionKernel.cu -o ./src/java/kernel/convolutionKernel.ptx -Xcompiler /wd4819 
echo Build forcomputeKernel.cu to forcomputeKernel.ptx & ^
nvcc -ptx ./src/java/kernel/forcomputeKernel.cu -o ./src/java/kernel/forcomputeKernel.ptx -Xcompiler /wd4819 
echo Build getHnew.cu to getHnew.ptx & ^
nvcc -ptx ./src/java/kernel/getHnew.cu -o ./src/java/kernel/getHnew.ptx -Xcompiler /wd4819 
echo Build getprojection.cu to getprojection.ptx & ^
nvcc -ptx ./src/java/kernel/getprojection.cu -o ./src/java/kernel/getprojection.ptx -Xcompiler /wd4819 
echo Build imshift.cu to imshift.ptx & ^
nvcc -ptx ./src/java/kernel/imshift.cu -o ./src/java/kernel/imshift.ptx -Xcompiler /wd4819 
echo Build multiply.cu to multiply.ptx & ^
nvcc -ptx ./src/java/kernel/multiply.cu -o ./src/java/kernel/multiply.ptx  -Xcompiler /wd4819 
echo Build setprojection.cu to setprojection.ptx & ^
nvcc -ptx ./src/java/kernel/setprojection.cu -o ./src/java/kernel/setprojection.ptx -Xcompiler /wd4819 
echo Build psfcomplex.cu to psfcomplex.ptx & ^
nvcc -ptx ./src/java/kernel/psfcomplex.cu -o ./src/java/kernel/psfcomplex.ptx -Xcompiler /wd4819 
echo Build imshiftap.cu to imshiftap.ptx & ^
nvcc -ptx ./src/java/kernel/imshiftap.cu -o ./src/java/kernel/imshiftap.ptx -Xcompiler /wd4819 
echo Build pixelbinning.cu to pixelbinning.ptx & ^
nvcc -ptx ./src/java/kernel/pixelbinning.cu -o ./src/java/kernel/pixelbinning.ptx -Xcompiler /wd4819 
echo Build getpsf.cu to getpsf.ptx & ^
nvcc -ptx ./src/java/kernel/getpsf.cu -o ./src/java/kernel/getpsf.ptx -Xcompiler /wd4819 
echo Build backcut.cu to backcut.ptx & ^
nvcc -ptx ./src/java/kernel/backcut.cu -o ./src/java/kernel/backcut.ptx -Xcompiler /wd4819 
echo Build backht.cu to backht.ptx & ^
nvcc -ptx ./src/java/kernel/backht.cu -o ./src/java/kernel/backht.ptx -Xcompiler /wd4819 
pause 