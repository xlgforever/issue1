#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <chrono>
#include <iostream>
#include <CL/opencl.h>

#include "bmp.h"
#include "sobelfilter_config.h"

#define KERNEL_NAME "SobelDetector"
#define IMAGE_SIZE  512
size_t localThreads[2] = {1, 1};           // {8, 1} is perfect

// This function will generate the data via a 
// CPU and we'll use this to verify our result against
// that computation done by the OpenCL/GPU
void goldenReferenceCPU(int width, 
                        int height, 
                        int pixelSize,
                        cl_uchar4* inputImage,
                        cl_uchar* dataForVerification) {
    // x-axis gradient mask 
    const int kernelx[][3] = 
    { 
        { 1, 2, 1},
        { 0, 0, 0},
        { -1,-2,-1}
    };

    // y-axis gradient mask
    const int kernely[][3] = 
    { 
        { 1, 0, -1},
        { 2, 0, -2},
        { 1, 0, -1}
    };

    int gx = 0;
    int gy = 0;

    // pointer to input image data
    cl_uchar *ptr = (cl_uchar*)malloc(width * height * pixelSize);
    memcpy(ptr, inputImage, width * height * pixelSize);

    // each pixel has 4 uchar components 
    int w = width * 4;

    int k = 1;

    // apply filter on each pixel (except boundary pixels)
    for(int i = 0; i < (int)(w * (height - 1)) ; i++) 
    {
        if(i < (k+1)*w - 4 && i >= 4 + k*w)
        {
            gx =  kernelx[0][0] * *(ptr + i - 4 - w)
                + kernelx[0][1] * *(ptr + i - w)
                + kernelx[0][2] * *(ptr + i + 4 - w)
                + kernelx[1][0] * *(ptr + i - 4)
                + kernelx[1][1] * *(ptr + i)
                + kernelx[1][2] * *(ptr + i + 4)
                + kernelx[2][0] * *(ptr + i - 4 + w)
                + kernelx[2][1] * *(ptr + i + w)
                + kernelx[2][2] * *(ptr + i + 4 + w);


            gy =  kernely[0][0] * *(ptr + i - 4 - w)
                + kernely[0][1] * *(ptr + i - w)
                + kernely[0][2] * *(ptr + i + 4 - w)
                + kernely[1][0] * *(ptr + i - 4)
                + kernely[1][1] * *(ptr + i)
                + kernely[1][2] * *(ptr + i + 4)
                + kernely[2][0] * *(ptr + i - 4 + w)
                + kernely[2][1] * *(ptr + i + w)
                + kernely[2][2] * *(ptr + i + 4 + w);

            float gx2 = pow((float)gx, 2);
            float gy2 = pow((float)gy, 2);


            *(dataForVerification + i) = (cl_uchar)(sqrt(gx2 + gy2) / 2.0);
        }

        // if reached at the end of its row then incr k
        if(i == (k + 1) * w - 5)
        {
            k++;
        }
    } 

    free(ptr);
}

int verifyResults(cl_uint width, 
                  cl_uint height, 
                  cl_uint pixelSize,
                  cl_uchar4* outputImageData,
                  cl_uchar* dataForVerification) {

        float *outputDevice = (float*)malloc(width * height * pixelSize);
        float *outputReference = (float*)malloc(width * height * pixelSize);
            // copy uchar data to float array
        for(int i = 0; i < (int)(width * height); i++)
        {
            outputDevice[i * 4 + 0] = outputImageData[i].s[0];
            outputDevice[i * 4 + 1] = outputImageData[i].s[1];
            outputDevice[i * 4 + 2] = outputImageData[i].s[2];
            outputDevice[i * 4 + 3] = outputImageData[i].s[3];

            outputReference[i * 4 + 0] = dataForVerification[i * 4 + 0];
            outputReference[i * 4 + 1] = dataForVerification[i * 4 + 1];
            outputReference[i * 4 + 2] = dataForVerification[i * 4 + 2];
            outputReference[i * 4 + 3] = dataForVerification[i * 4 + 3];
        }

    free(outputDevice);
    free(outputReference);
    return 1;
}

int writeImage(cl_uint width,
               cl_uint height,
               cl_uint pixelSize,
               uchar4* pixelData,
               cl_uchar4* outputImageData,
               BitMap* inputBitMap,
               const char* outputImageName) {
    // copy output image data back to original pixel data
    
    memcpy(inputBitMap->pixels_, outputImageData, width * height * pixelSize);

    // write the output bmp file
    if(!writeA(outputImageName,inputBitMap))
    {
        printf("Failed to write output image!");
        return FAILURE;
    }
    printf("Write Image: %s is saved! \n\n", outputImageName);
    return SUCCESS;
}

void loadProgramSource(const char** files,
                       size_t length,
                       char** buffer,
                       size_t* sizes) {
    /* Read each source file (*.cl) and store the contents into a temporary datastore */
    for(size_t i=0; i < length; i++) {
        FILE* file = fopen(files[i], "r");
        if(file == NULL) {
            perror("Couldn't read the program file");
            exit(1);
        }
        fseek(file, 0, SEEK_END);
        sizes[i] = ftell(file);
        rewind(file); // reset the file pointer so that 'fread' reads from the front
        buffer[i] = (char*)malloc(sizes[i]+1);
        buffer[i][sizes[i]] = '\0';
        fread(buffer[i], sizeof(char), sizes[i], file);
        fclose(file);
    }
}

int
decipherEvent(cl_event* event) {
    cl_int status = CL_SUCCESS;
    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE) {
        status = clGetEventInfo(*event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &eventStatus, NULL);
    }
    clReleaseEvent(*event);
    return 0;
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;
  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
    return -1;
  }
  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);
  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);
  fclose(fp);
  return 0;
}



int main(int argc, char** argv) {
    /* OpenCL 1.1 data structures */
    cl_platform_id* platforms;
    cl_program program;
    cl_device_id device;
    cl_kernel kernel;
    cl_context context;
    cl_command_queue queue;
    cl_mem inputImageBuffer;
    cl_mem outputImageBuffer; 

    /* OpenCL 1.1 scalar data types */
    cl_uint numOfPlatforms;
    cl_int  error;
    cl_uint pixelSize = sizeof(uchar4);
    uchar4* pixelData = NULL;
    cl_uint width;
    cl_uint height;
    cl_uchar* output = NULL;
    BitMap inputBitMap;

    cl_uchar4* inputImageData = NULL;
    cl_uchar4* outputImageData = NULL;

	{
	    // load input bitmap image 
        char imagename[64]; 
        sprintf(imagename,"InputImage%d.bmp",IMAGE_SIZE);
	    load(imagename, &inputBitMap);
        std::cout<< "Load InputImage\n" << &inputBitMap << std::endl;
	
	    // error if image did not load
	    if(!isLoaded(&inputBitMap))
	    {
	        printf("Failed to load input image!\n");
	        return FAILURE;
	    }
	
	
	    // get width and height of input image 
	    height = getHeight(&inputBitMap);
	    width = getWidth(&inputBitMap);
        printf("inputimage size is [%d,%d]!\n",height,width);
	    // allocate memory for input & output image data 
	    inputImageData  = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
	    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
	    // initializa the Image data to NULL
	    memset(outputImageData, 0, width * height * pixelSize);
	    // get the pointer to pixel data 
        pixelData = (uchar4*)malloc(width * height * sizeof(cl_uchar4));
	    memcpy(pixelData, getPixels(&inputBitMap), width * height * pixelSize);
	    // Copy pixel data into inputImageData 
	    memcpy(inputImageData, pixelData, width * height * pixelSize);
	    // allocate memory for verification output
	    output = (cl_uchar*)malloc(width * height * pixelSize);
	    // initialize the data to NULL 
	    memset(output, 0, width * height * pixelSize);
    }


    error = clGetPlatformIDs(0, NULL, &numOfPlatforms);
    if(error != CL_SUCCESS) {
        perror("Unable to find any OpenCL platforms");
        exit(1);
    }
    printf("find %d platforms\n", numOfPlatforms);
    
    platforms = (cl_platform_id*) alloca(sizeof(cl_platform_id) * numOfPlatforms);
    printf("Number of OpenCL platforms found: %d\n", numOfPlatforms);
    
    //error = clGetPlatformIDs(numOfPlatforms, platforms, NULL);
    error = clGetPlatformIDs(1, platforms, NULL);
    if(error != CL_SUCCESS) {
        perror("Unable to find any OpenCL platforms");
        exit(1);
    }

    for(cl_uint i = 0; i < numOfPlatforms; i++ ) {
        // Get the GPU device
        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if(error != CL_SUCCESS) {
            perror("Can't locate a OpenCL compliant device i.e. GPU");
            printf("error is %d\n", error);
            exit(1);
        }
        printf("error is %d\n", error);
        
        size_t pSize_Vendor = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, NULL, NULL, &pSize_Vendor);
        cl_uint* cu = (cl_uint*)malloc(sizeof(cl_uint)*pSize_Vendor);
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, pSize_Vendor, cu, NULL);
        std::cout << "CL_DEVICE_MAX_COMPUTE_UINTS\t " << *cu << std::endl; //4096

        size_t pSize_VendorS = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, NULL, NULL, &pSize_VendorS);
        cl_uint* cuS = (cl_uint*)malloc(sizeof(cl_uint)*pSize_Vendor);
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, pSize_VendorS, cuS, NULL);
        std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE\t " << *cuS << std::endl<< std::endl; //4096


        /* Create a context */
        context = clCreateContext(NULL, 1, &device, NULL, NULL, &error);
        if(error != CL_SUCCESS) {
            perror("Can't create a valid OpenCL context");
            exit(1);
        }
        
        /* Load the two source files into temporary datastores */
        const char *file_names[] = {"sobel_detector.cl"};
        const int NUMBER_OF_FILES = 1;
        char* buffer[NUMBER_OF_FILES];
        size_t sizes[NUMBER_OF_FILES];
        //loadProgramSource(file_names, NUMBER_OF_FILES, buffer, sizes);
        
        // read kernel binary from file  
        uint8_t *kernel_bin = NULL;
        size_t kernel_size;
        cl_int binary_status;
        cl_int _err = CL_INVALID_VALUE;
        if (0 != read_kernel_file("kernel.pocl", &kernel_bin, &kernel_size))
            return -1;
        std::cout<< "kernel_size\t\t" << kernel_size <<std::endl;


        program = clCreateProgramWithBinary(
            context, 1, &device, &kernel_size, (const uint8_t**)&kernel_bin, &binary_status, &_err);
        printf("clCreateProgramWithBinary  error is %d and binary_status is %d \n",_err,binary_status);
        if (program == NULL) {
            return -1;
        }

        /* Build OpenCL program object and dump the error message, if any */
        char *program_log;
        const char options[] = "";
        size_t log_size;

        error = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
        printf("clBuildProgram  error is %d\n",error);
	    if(error != CL_SUCCESS) {
            // If there's an error whilst building the program, dump the log
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            program_log = (char*) malloc(log_size+1);
            program_log[log_size] = '\0';
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                  log_size+1, program_log, NULL);
            printf("\n=== ERROR ===\n\n%s\n=============\n", program_log);
            free(program_log);
            exit(1);
	    }

        kernel = clCreateKernel(program, KERNEL_NAME, &error);
        if (kernel == NULL) {
            printf("clCreateKernel  fialed\n");
            return -1;
        }

        queue = clCreateCommandQueue(context, device, 0, &error);
        printf("clCreateCommandQueue  error is %d\n\n",error);

        //printf("clEnqueueWriteBuffer  \n");
        //clEnqueueWriteBuffer(queue, 
        //                    inputImageBuffer, CL_TRUE, 0, 
        //                    width * height * pixelSize, 
        //                    inputImageData, 
        //                    0, NULL, NULL);


        inputImageBuffer = clCreateBuffer(context,
                                          CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                          width * height * pixelSize,inputImageData,
                                          &error);printf("inputImageBuffer is finished! \n\n");
        outputImageBuffer = clCreateBuffer(context,
                                           CL_MEM_WRITE_ONLY|CL_MEM_COPY_HOST_PTR,
                                           width * height * pixelSize,outputImageData,
                                           &error);printf("outputImageBuffer is finished! \n\n");    

        clSetKernelArg(kernel, 0, sizeof(cl_mem),(void*)&inputImageBuffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem),(void*)&outputImageBuffer);
         


        size_t globalThreads[] = {width*4, height};//{512,512}
        //size_t localThreads[]  = {sizeX, sizeY}; 
        //{sizeX = GROUP_SIZE,1} 工作组尺寸
		cl_event exeEvt; 

        auto time_start = std::chrono::high_resolution_clock::now();
		error = clEnqueueNDRangeKernel(queue,
		                               kernel,
		                               2,                            // 2-D data
		                               NULL, 
                                       globalThreads,                // image size 
                                       localThreads,                 // work-group size
                                       0, NULL, &exeEvt);
        printf("clEnqueueNDRangeKernel  error is %d\n\n",error);
        auto time_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count()/1000.0;
        std::cout << "execution time is:\t" << elapsed << std::endl;

		clWaitForEvents(1, &exeEvt);
		if(error != CL_SUCCESS) {
			printf("Kernel execution failure!\n");
			exit(-22);
		}	
        clReleaseEvent(exeEvt);
 
        printf("clEnqueueReadBuffer  \n");
        clEnqueueReadBuffer(queue,
                            outputImageBuffer,CL_TRUE,0,
                            width * height * pixelSize,
                            outputImageData,
                            0,NULL,NULL);

        char outname[64]; 
        sprintf(outname,"OutputImage%d_s%d_%d_t%.3f.bmp",
                IMAGE_SIZE,localThreads[0],localThreads[1],elapsed);

        writeImage(width, height, pixelSize, pixelData, 
                    outputImageData, &inputBitMap, outname);
        
        /* Clean up */
        for(i=0; i< NUMBER_OF_FILES; i++) { free(buffer[i]); }
        clReleaseProgram(program);
        clReleaseContext(context);
        clReleaseMemObject(inputImageBuffer);
        clReleaseMemObject(outputImageBuffer);
    }
    
    free(inputImageData);free(outputImageData);free(output);
}
