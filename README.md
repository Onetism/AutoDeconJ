# AutoDeconJ: a GPU-accelerated ImageJ plugin for 3D light-field deconvolution with optimal iteration numbers predicting

**[Paer](https://nerfmm.active.vision/)|[User maunual](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/PAP/10.1093_bioinformatics_btac760/2/btac760_supplementary_data.zip?Expires=1673249503&Signature=fEoHQOkTfkO2u5c5Qcmow-vzfD3EfLMgIVVsEC-Joo4aD8CnvckZ0FqJSpqpkENz2lvqkTsBIcgJZS5rdE4SMRGGm5UsYSenrchkAN0ttcdtJmxb~bTK8A3qFNG1dhO~0lM5XNxl1mqk-s0C9L~JHaHdOK3bFQSyiVHGz~4xyP4tGQ8aADkJmI0Ko3adPzSyxl5Fp8AooPlAbj3GBVsF4lT0pXW-DcxuJ6eBRBVID~mBB5pWlt9zXwcgC2eaS3FOqxUcTLfE8cTxTkzfK5kkfLgoReWjBMOlb7mVEHwxLX0KQn2uyNxpCqsFNvyn3AK3GM1amOwiatlYxbax8GXydA__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA)**

Changqing Su, Yuhan Gao, You Zhou, Yaoqi Sun, Chenggang Yan, Haibing Yin, Bo Xiong


Light-field microscopy (LFM) is a compact solution to high-speed 3D fluorescence imaging. Usually, we need to do 3D deconvolution to the captured raw data. Although there are deep neural network methods that can accelerate the reconstruction process, the model is not universally applicable for all system parameters. Here, we develop AutoDeconJ, a GPU-accelerated ImageJ plugin for 4.4× faster and more accurate deconvolution of LFM data. We further propose an image quality metric for the deconvolution process, aiding in automatically determining the optimal number of iterations with higher reconstruction accuracy and fewer artifacts.

<img src="https://github.com/Onetism/AutoDeconJ/blob/main/AutoDeconJ.png" width="600" /> 

# Installation
## Cloning the repository

Create a local clone of this project by calling

    git clone https://github.com/Onetism/AutoDeconJ.git
    

## Build PTX
AutoDeconJ requires the NVIDIA cards support by CUDA8.0 or later. See https://developer.nvidia.com for more details about
CUDA. Please make sure that CUDA is properly installed and the path enviroment is configured correctly.
In order to build the ptx file for java corresponding to the version of CUDA, change into the root directory of the 
project and execute

On linux

    ./build_cuda.sh

On Windows

    ./build_cuda.bat

## Building the plugin JAR and install in ImageJ

The cloned project can be opened and edited in any IDE (e.g. Eclipse, 
NetBeans, IntelliJ...). However, the preferred way to build the final 
plugin JAR is via [Apache Maven](https://maven.apache.org/).

In order to build the plugin JAR, change into the root directory of the 
project and execute

    mvn clean package
    
Note that the resulting JAR file has a name that is different from the 
default name that Maven would assign to it: In order to properly be recognized 
as an ImageJ plugin JAR, it is named `AutoDeconJ_Plugin-jar-with-dependencies.jar`.
Copy the resulting `/target/AutoDeconJ_Plugin-jar-with-dependencies.jar` file into 
the `Fiji.app/plugins` directory of your Fiji/ImageJ installation. Of course, 
`AutoDeconJ_Plugin-jar-with-dependencies.jar` is also a execution file, you can execute
directly.

## Using the plugin

After the JAR files for the plugin have been added, it may be used inside 
ImageJ: Start ImageJ, load an image, and select 

    Plugins>AutoDeconJ, "Run Auto_LF_Deconvolution...", Auto_LF_Deconvolution("run")
    Plugins>AutoDeconJ, "Run ImageRectification...", ImageRectification("run")
    
from the menu bar. 

## Test Data
There is a test images in `/test/resources`, corresponding to the default parameters in the program: 
objective magnification 40, NA 0.8, microlens pitch size 150 um, microlens focal length 3000um, wavelength 520um, 
n 1.0, OSR 3, Nnum 15, z-spacing 2um, z-max 0um, and z-min -26um.

---

## Citation
```
@article{AutoDeconJ,
    author = {Su, Changqing and Gao, Yuhan and Zhou, You and Sun, Yaoqi and Yan, Chenggang and Yin, Haibing and Xiong, Bo},
    title = "{AutoDeconJ: a GPU-accelerated ImageJ plugin for 3D light-field deconvolution with optimal iteration numbers predicting}",
    journal = {Bioinformatics},
    year = {2022},
    month = {11},
    issn = {1367-4803},
}
```