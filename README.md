# MultiGPU_DeconJ

A basic example project for a simple ImageJ plugin that uses MultiGPU.

## About ImageJ

This project shows how to create an ImageJ1 plugin. As of 04/2018, ImageJ is 
in the process of being updated and refactored, to be released as ImageJ2. 
The new version is built on a different infrastructure for handling plugins. 
However, ImageJ plugins are still supposed to work in ImageJ2. For details 
about the differences between ImageJ1 and ImageJ2, refer to the
[ImageJ FAQ](https://imagej.net/Frequently_Asked_Questions).

Currently, the recommended way to obtain an up-to-date ImageJ installation
is via [Fiji](http://imagej.net/Fiji). Just download the distribution for
your operating system and unpack it. Fiji/ImageJ will then be available
in the `Fiji.app` directory.


## Cloning the repository

Create a local clone of this project by calling

    git clone https://github.com/Onetism/MultiGPU_DeconJ.git
    

## Building the plugin JAR

The cloned project can be opened and edited in any IDE (e.g. Eclipse, 
NetBeans, IntelliJ...). However, the preferred way to build the final 
plugin JAR is via [Apache Maven](https://maven.apache.org/).

In order to build the plugin JAR, change into the root directory of the 
project and execute

    mvn clean package
    
Note that the resulting JAR file has a name that is different from the 
default name that Maven would assign to it: In order to properly be recognized 
as an ImageJ plugin JAR, it is named `MultiGPU_DeconJ_ImageJ_Plugin-jar-with-dependencies.jar`.
Copy the resulting `/target/MultiGPU_DeconJ_ImageJ_Plugin-jar-with-dependencies.jar` file into 
the `Fiji.app/plugins` directory of your Fiji/ImageJ installation. Of course, 
`MultiGPU_DeconJ_ImageJ_Plugin-jar-with-dependencies.jar` is also a execution file, you can execute
directly.


## Adding the dependencies

The plugin JAR has dependencies to other JAR files. The following command 
can be used to collect all the dependencies: 

    mvn dependency:copy-dependencies 

This will copy all required dependencies into the `/dependency` subdirectory
of the project. Copy these JARs into the `Fiji.app/jars` directory of your 
Fiji/ImageJ installation.

**Note:** These dependencies will only include the platform-specific JAR files.
For example, on Windows, they will include the 
`jcuda-natives-0.9.0d-windows-x86_64.jar` file. 
In order to support other operating systems, the corresponding JAR files
(e.g. `jcuda-natives-0.9.0d-linux-x86_64.jar`) will have to be added 
manually for the respective Fiji/ImageJ installation.


## Using the plugin

After the JAR files for the plugin have been added, it may be used inside 
ImageJ: Start ImageJ, load an image, and select 

    Plugins>multiGPU_DeconJ, "Run Auto_LF_Deconvolution...", Auto_LF_Deconvolution("run")
    Plugins>multiGPU_DeconJ, "Run ImageRectification...", ImageRectification("run")
    
from the menu bar. 

---

## Building your own plugin based on the example

Note that the plugin that is shown in this project is very simple and
minimalistic. In order to develop more complex plugins, refer to the
[ImageJ Wiki about Plugins](https://imagej.net/Plugins) and the general
[ImageJ Development Documentation](https://imagej.net/Development).

In order to start first experiments, you may use this project as a 
template. The project directory structure and the most relevant files 
are summarized here:

    jcuda-imagej-example/
        pom.xml
        src/
          main/
            java/
              JCuda_ImageJ_Example_Plugin.java
            resources/
              JCudaImageJExampleKernel.cu
              plugins.config


The following is a short summary of the information that is contained in 
each of these files, and how it may be changed for your own plugin:

### The `pom.xml`

The `pom.xml` contains the Maven names for the project. For your own
plugin, you should adjust the `<groupId>` and `<artifactId>` in the 
`pom.xml` file. For example:
   
    <groupId>com.example</groupId>
    <artifactId>my-imagej-plugin</artifactId>
   
The `pom.xml` also defines the name of the resulting plugin JAR file.
For your own plugin, you could change it as shown in this example:
     
    <properties>
        <imagej.plugin.name>My_ImageJ_Plugin</imagej.plugin.name>
        ...
    </properties>
    
**Note:** Due to some constraints of ImageJ, the plugin name must contain 
an `'_'` underscore character!


### The `plugins.config` file

This file basically defines where in the *Plugins* menu of ImageJ the 
plugin will appear. The relevant entry for the JCuda example plugin is this:

    Plugins>multiGPU_DeconJ, "Run Auto_LF_Deconvolution...", Auto_LF_Deconvolution("run")
    Plugins>multiGPU_DeconJ, "Run ImageRectification...", ImageRectification("run")
    
The first part defines the menu structure. The last part, `Auto_LF_Deconvolution`,
defines the *class name* of the plugin implementation.

 

    
