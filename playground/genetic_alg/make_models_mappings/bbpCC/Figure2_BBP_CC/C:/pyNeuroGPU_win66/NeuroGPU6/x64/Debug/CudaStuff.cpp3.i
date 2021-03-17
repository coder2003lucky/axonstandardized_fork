#line 1 "x64/Debug/CudaStuff.cudafe2.gpu"
#line 66 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
struct HMat;
#line 111 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
struct Stim;
#line 126 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
struct Sim;
#line 194 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
typedef unsigned long long size_t;
#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"





































#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"

























































#line 59 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"



        








   


#line 75 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"
        



































































#line 144 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"














#line 161 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"






#line 168 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"




#line 173 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"










#line 185 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"













        














#line 214 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"
























#line 239 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"


#line 242 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\host_defines.h"
#line 39 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"





typedef  unsigned long long __texture_type__;
typedef  unsigned long long __surface_type__;



#line 50 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"













#line 64 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"


























































































































































































































#line 283 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"

#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\builtin_types.h"























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_types.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
















































































































































































































































#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
#line 54 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_types.h"







enum  cudaRoundMode
{
    cudaRoundNearest,
    cudaRoundZero,
    cudaRoundPosInf,
    cudaRoundMinInf
};

#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_types.h"
#line 57 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\builtin_types.h"


#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
















































































































































































































































#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
#line 54 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"



























































































#line 146 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"










enum  cudaError
{
    




    cudaSuccess                           =      0,
  
    



    cudaErrorMissingConfiguration         =      1,
  
    



    cudaErrorMemoryAllocation             =      2,
  
    



    cudaErrorInitializationError          =      3,
  
    







    cudaErrorLaunchFailure                =      4,
  
    






    cudaErrorPriorLaunchFailure           =      5,
  
    








    cudaErrorLaunchTimeout                =      6,
  
    






    cudaErrorLaunchOutOfResources         =      7,
  
    



    cudaErrorInvalidDeviceFunction        =      8,
  
    






    cudaErrorInvalidConfiguration         =      9,
  
    



    cudaErrorInvalidDevice                =     10,
  
    



    cudaErrorInvalidValue                 =     11,
  
    



    cudaErrorInvalidPitchValue            =     12,
  
    



    cudaErrorInvalidSymbol                =     13,
  
    


    cudaErrorMapBufferObjectFailed        =     14,
  
    


    cudaErrorUnmapBufferObjectFailed      =     15,
  
    



    cudaErrorInvalidHostPointer           =     16,
  
    



    cudaErrorInvalidDevicePointer         =     17,
  
    



    cudaErrorInvalidTexture               =     18,
  
    



    cudaErrorInvalidTextureBinding        =     19,
  
    




    cudaErrorInvalidChannelDescriptor     =     20,
  
    



    cudaErrorInvalidMemcpyDirection       =     21,
  
    







    cudaErrorAddressOfConstant            =     22,
  
    






    cudaErrorTextureFetchFailed           =     23,
  
    






    cudaErrorTextureNotBound              =     24,
  
    






    cudaErrorSynchronizationError         =     25,
  
    



    cudaErrorInvalidFilterSetting         =     26,
  
    



    cudaErrorInvalidNormSetting           =     27,
  
    





    cudaErrorMixedDeviceExecution         =     28,
  
    




    cudaErrorCudartUnloading              =     29,
  
    


    cudaErrorUnknown                      =     30,

    





    cudaErrorNotYetImplemented            =     31,
  
    






    cudaErrorMemoryValueTooLarge          =     32,
  
    




    cudaErrorInvalidResourceHandle        =     33,
  
    





    cudaErrorNotReady                     =     34,
  
    




    cudaErrorInsufficientDriver           =     35,
  
    










    cudaErrorSetOnActiveProcess           =     36,
  
    



    cudaErrorInvalidSurface               =     37,
  
    



    cudaErrorNoDevice                     =     38,
  
    



    cudaErrorECCUncorrectable             =     39,
  
    


    cudaErrorSharedObjectSymbolNotFound   =     40,
  
    


    cudaErrorSharedObjectInitFailed       =     41,
  
    



    cudaErrorUnsupportedLimit             =     42,
  
    



    cudaErrorDuplicateVariableName        =     43,
  
    



    cudaErrorDuplicateTextureName         =     44,
  
    



    cudaErrorDuplicateSurfaceName         =     45,
  
    







    cudaErrorDevicesUnavailable           =     46,
  
    


    cudaErrorInvalidKernelImage           =     47,
  
    





    cudaErrorNoKernelImageForDevice       =     48,
  
    










    cudaErrorIncompatibleDriverContext    =     49,
      
    




    cudaErrorPeerAccessAlreadyEnabled     =     50,
    
    




    cudaErrorPeerAccessNotEnabled         =     51,
    
    



    cudaErrorDeviceAlreadyInUse           =     54,

    




    cudaErrorProfilerDisabled             =     55,

    





    cudaErrorProfilerNotInitialized       =     56,

    




    cudaErrorProfilerAlreadyStarted       =     57,

    




     cudaErrorProfilerAlreadyStopped       =    58,

    





    cudaErrorAssert                        =    59,
  
    




    cudaErrorTooManyPeers                 =     60,
  
    



    cudaErrorHostMemoryAlreadyRegistered  =     61,
        
    



    cudaErrorHostMemoryNotRegistered      =     62,

    


    cudaErrorOperatingSystem              =     63,

    



    cudaErrorPeerAccessUnsupported        =     64,

    




    cudaErrorLaunchMaxDepthExceeded       =     65,

    





    cudaErrorLaunchFileScopedTex          =     66,

    





    cudaErrorLaunchFileScopedSurf         =     67,

    












    cudaErrorSyncDepthExceeded            =     68,

    









    cudaErrorLaunchPendingCountExceeded   =     69,
    
    


    cudaErrorNotPermitted                 =     70,

    



    cudaErrorNotSupported                 =     71,

    






    cudaErrorHardwareStackError           =     72,

    





    cudaErrorIllegalInstruction           =     73,

    






    cudaErrorMisalignedAddress            =     74,

    








    cudaErrorInvalidAddressSpace          =     75,

    





    cudaErrorInvalidPc                    =     76,

    





    cudaErrorIllegalAddress               =     77,

    



    cudaErrorInvalidPtx                   =     78,

    


    cudaErrorInvalidGraphicsContext       =     79,

    



    cudaErrorNvlinkUncorrectable          =     80,

    


    cudaErrorStartupFailure               =   0x7f,

    





    cudaErrorApiFailureBase               =  10000
};




enum  cudaChannelFormatKind
{
    cudaChannelFormatKindSigned           =   0,      
    cudaChannelFormatKindUnsigned         =   1,      
    cudaChannelFormatKindFloat            =   2,      
    cudaChannelFormatKindNone             =   3       
};




struct  cudaChannelFormatDesc
{
    int                        x; 
    int                        y; 
    int                        z; 
    int                        w; 
    enum cudaChannelFormatKind f; 
};




typedef struct cudaArray *cudaArray_t;




typedef const struct cudaArray *cudaArray_const_t;

struct cudaArray;




typedef struct cudaMipmappedArray *cudaMipmappedArray_t;




typedef const struct cudaMipmappedArray *cudaMipmappedArray_const_t;

struct cudaMipmappedArray;




enum  cudaMemoryType
{
    cudaMemoryTypeHost   = 1, 
    cudaMemoryTypeDevice = 2  
};




enum  cudaMemcpyKind
{
    cudaMemcpyHostToHost          =   0,      
    cudaMemcpyHostToDevice        =   1,      
    cudaMemcpyDeviceToHost        =   2,      
    cudaMemcpyDeviceToDevice      =   3,      
    cudaMemcpyDefault             =   4       
};






struct  cudaPitchedPtr
{
    void   *ptr;      
    size_t  pitch;    
    size_t  xsize;    
    size_t  ysize;    
};






struct  cudaExtent
{
    size_t width;     
    size_t height;    
    size_t depth;     
};






struct  cudaPos
{
    size_t x;     
    size_t y;     
    size_t z;     
};




struct  cudaMemcpy3DParms
{
    cudaArray_t            srcArray;  
    struct cudaPos         srcPos;    
    struct cudaPitchedPtr  srcPtr;    
  
    cudaArray_t            dstArray;  
    struct cudaPos         dstPos;    
    struct cudaPitchedPtr  dstPtr;    
  
    struct cudaExtent      extent;    
    enum cudaMemcpyKind    kind;      
};




struct  cudaMemcpy3DPeerParms
{
    cudaArray_t            srcArray;  
    struct cudaPos         srcPos;    
    struct cudaPitchedPtr  srcPtr;    
    int                    srcDevice; 
  
    cudaArray_t            dstArray;  
    struct cudaPos         dstPos;    
    struct cudaPitchedPtr  dstPtr;    
    int                    dstDevice; 
  
    struct cudaExtent      extent;    
};




struct cudaGraphicsResource;




enum  cudaGraphicsRegisterFlags
{
    cudaGraphicsRegisterFlagsNone             = 0,  
    cudaGraphicsRegisterFlagsReadOnly         = 1,   
    cudaGraphicsRegisterFlagsWriteDiscard     = 2,  
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,  
    cudaGraphicsRegisterFlagsTextureGather    = 8   
};




enum  cudaGraphicsMapFlags
{
    cudaGraphicsMapFlagsNone         = 0,  
    cudaGraphicsMapFlagsReadOnly     = 1,  
    cudaGraphicsMapFlagsWriteDiscard = 2   
};




enum  cudaGraphicsCubeFace 
{
    cudaGraphicsCubeFacePositiveX = 0x00, 
    cudaGraphicsCubeFaceNegativeX = 0x01, 
    cudaGraphicsCubeFacePositiveY = 0x02, 
    cudaGraphicsCubeFaceNegativeY = 0x03, 
    cudaGraphicsCubeFacePositiveZ = 0x04, 
    cudaGraphicsCubeFaceNegativeZ = 0x05  
};




enum  cudaResourceType
{
    cudaResourceTypeArray          = 0x00, 
    cudaResourceTypeMipmappedArray = 0x01, 
    cudaResourceTypeLinear         = 0x02, 
    cudaResourceTypePitch2D        = 0x03  
};




enum  cudaResourceViewFormat
{
    cudaResViewFormatNone                      = 0x00, 
    cudaResViewFormatUnsignedChar1             = 0x01, 
    cudaResViewFormatUnsignedChar2             = 0x02, 
    cudaResViewFormatUnsignedChar4             = 0x03, 
    cudaResViewFormatSignedChar1               = 0x04, 
    cudaResViewFormatSignedChar2               = 0x05, 
    cudaResViewFormatSignedChar4               = 0x06, 
    cudaResViewFormatUnsignedShort1            = 0x07, 
    cudaResViewFormatUnsignedShort2            = 0x08, 
    cudaResViewFormatUnsignedShort4            = 0x09, 
    cudaResViewFormatSignedShort1              = 0x0a, 
    cudaResViewFormatSignedShort2              = 0x0b, 
    cudaResViewFormatSignedShort4              = 0x0c, 
    cudaResViewFormatUnsignedInt1              = 0x0d, 
    cudaResViewFormatUnsignedInt2              = 0x0e, 
    cudaResViewFormatUnsignedInt4              = 0x0f, 
    cudaResViewFormatSignedInt1                = 0x10, 
    cudaResViewFormatSignedInt2                = 0x11, 
    cudaResViewFormatSignedInt4                = 0x12, 
    cudaResViewFormatHalf1                     = 0x13, 
    cudaResViewFormatHalf2                     = 0x14, 
    cudaResViewFormatHalf4                     = 0x15, 
    cudaResViewFormatFloat1                    = 0x16, 
    cudaResViewFormatFloat2                    = 0x17, 
    cudaResViewFormatFloat4                    = 0x18, 
    cudaResViewFormatUnsignedBlockCompressed1  = 0x19, 
    cudaResViewFormatUnsignedBlockCompressed2  = 0x1a, 
    cudaResViewFormatUnsignedBlockCompressed3  = 0x1b, 
    cudaResViewFormatUnsignedBlockCompressed4  = 0x1c, 
    cudaResViewFormatSignedBlockCompressed4    = 0x1d, 
    cudaResViewFormatUnsignedBlockCompressed5  = 0x1e, 
    cudaResViewFormatSignedBlockCompressed5    = 0x1f, 
    cudaResViewFormatUnsignedBlockCompressed6H = 0x20, 
    cudaResViewFormatSignedBlockCompressed6H   = 0x21, 
    cudaResViewFormatUnsignedBlockCompressed7  = 0x22  
};




struct  cudaResourceDesc {
	enum cudaResourceType resType;             
	
	union {
		struct {
			cudaArray_t array;                 
		} array;
        struct {
            cudaMipmappedArray_t mipmap;       
        } mipmap;
		struct {
			void *devPtr;                      
			struct cudaChannelFormatDesc desc; 
			size_t sizeInBytes;                
		} linear;
		struct {
			void *devPtr;                      
			struct cudaChannelFormatDesc desc; 
			size_t width;                      
			size_t height;                     
			size_t pitchInBytes;               
		} pitch2D;
	} res;
};




struct  cudaResourceViewDesc
{
    enum cudaResourceViewFormat format;           
    size_t                      width;            
    size_t                      height;           
    size_t                      depth;            
    unsigned int                firstMipmapLevel; 
    unsigned int                lastMipmapLevel;  
    unsigned int                firstLayer;       
    unsigned int                lastLayer;        
};




struct  cudaPointerAttributes
{
    



    enum cudaMemoryType memoryType;

    








    int device;

    



    void *devicePointer;

    



    void *hostPointer;

    


    int isManaged;
};




struct  cudaFuncAttributes
{
   




   size_t sharedSizeBytes;

   



   size_t constSizeBytes;

   


   size_t localSizeBytes;

   




   int maxThreadsPerBlock;

   


   int numRegs;

   




   int ptxVersion;

   




   int binaryVersion;

   



   int cacheModeCA;
};




enum  cudaFuncCache
{
    cudaFuncCachePreferNone   = 0,    
    cudaFuncCachePreferShared = 1,    
    cudaFuncCachePreferL1     = 2,    
    cudaFuncCachePreferEqual  = 3     
};





enum  cudaSharedMemConfig
{
    cudaSharedMemBankSizeDefault   = 0,
    cudaSharedMemBankSizeFourByte  = 1,
    cudaSharedMemBankSizeEightByte = 2
};




enum  cudaComputeMode
{
    cudaComputeModeDefault          = 0,  
    cudaComputeModeExclusive        = 1,  
    cudaComputeModeProhibited       = 2,  
    cudaComputeModeExclusiveProcess = 3   
};




enum  cudaLimit
{
    cudaLimitStackSize                    = 0x00, 
    cudaLimitPrintfFifoSize               = 0x01, 
    cudaLimitMallocHeapSize               = 0x02, 
    cudaLimitDevRuntimeSyncDepth          = 0x03, 
    cudaLimitDevRuntimePendingLaunchCount = 0x04  
};




enum  cudaMemoryAdvise
{
    cudaMemAdviseSetReadMostly          = 1, 
    cudaMemAdviseUnsetReadMostly        = 2, 
    cudaMemAdviseSetPreferredLocation   = 3, 
    cudaMemAdviseUnsetPreferredLocation = 4, 
    cudaMemAdviseSetAccessedBy          = 5, 
    cudaMemAdviseUnsetAccessedBy        = 6  
};




enum  cudaMemRangeAttribute
{
    cudaMemRangeAttributeReadMostly           = 1, 
    cudaMemRangeAttributePreferredLocation    = 2, 
    cudaMemRangeAttributeAccessedBy           = 3, 
    cudaMemRangeAttributeLastPrefetchLocation = 4  
};




enum  cudaOutputMode
{
    cudaKeyValuePair    = 0x00, 
    cudaCSV             = 0x01  
};




enum  cudaDeviceAttr
{
    cudaDevAttrMaxThreadsPerBlock             = 1,  
    cudaDevAttrMaxBlockDimX                   = 2,  
    cudaDevAttrMaxBlockDimY                   = 3,  
    cudaDevAttrMaxBlockDimZ                   = 4,  
    cudaDevAttrMaxGridDimX                    = 5,  
    cudaDevAttrMaxGridDimY                    = 6,  
    cudaDevAttrMaxGridDimZ                    = 7,  
    cudaDevAttrMaxSharedMemoryPerBlock        = 8,  
    cudaDevAttrTotalConstantMemory            = 9,  
    cudaDevAttrWarpSize                       = 10, 
    cudaDevAttrMaxPitch                       = 11, 
    cudaDevAttrMaxRegistersPerBlock           = 12, 
    cudaDevAttrClockRate                      = 13, 
    cudaDevAttrTextureAlignment               = 14, 
    cudaDevAttrGpuOverlap                     = 15, 
    cudaDevAttrMultiProcessorCount            = 16, 
    cudaDevAttrKernelExecTimeout              = 17, 
    cudaDevAttrIntegrated                     = 18, 
    cudaDevAttrCanMapHostMemory               = 19, 
    cudaDevAttrComputeMode                    = 20, 
    cudaDevAttrMaxTexture1DWidth              = 21, 
    cudaDevAttrMaxTexture2DWidth              = 22, 
    cudaDevAttrMaxTexture2DHeight             = 23, 
    cudaDevAttrMaxTexture3DWidth              = 24, 
    cudaDevAttrMaxTexture3DHeight             = 25, 
    cudaDevAttrMaxTexture3DDepth              = 26, 
    cudaDevAttrMaxTexture2DLayeredWidth       = 27, 
    cudaDevAttrMaxTexture2DLayeredHeight      = 28, 
    cudaDevAttrMaxTexture2DLayeredLayers      = 29, 
    cudaDevAttrSurfaceAlignment               = 30, 
    cudaDevAttrConcurrentKernels              = 31, 
    cudaDevAttrEccEnabled                     = 32, 
    cudaDevAttrPciBusId                       = 33, 
    cudaDevAttrPciDeviceId                    = 34, 
    cudaDevAttrTccDriver                      = 35, 
    cudaDevAttrMemoryClockRate                = 36, 
    cudaDevAttrGlobalMemoryBusWidth           = 37, 
    cudaDevAttrL2CacheSize                    = 38, 
    cudaDevAttrMaxThreadsPerMultiProcessor    = 39, 
    cudaDevAttrAsyncEngineCount               = 40, 
    cudaDevAttrUnifiedAddressing              = 41,     
    cudaDevAttrMaxTexture1DLayeredWidth       = 42, 
    cudaDevAttrMaxTexture1DLayeredLayers      = 43, 
    cudaDevAttrMaxTexture2DGatherWidth        = 45, 
    cudaDevAttrMaxTexture2DGatherHeight       = 46, 
    cudaDevAttrMaxTexture3DWidthAlt           = 47, 
    cudaDevAttrMaxTexture3DHeightAlt          = 48, 
    cudaDevAttrMaxTexture3DDepthAlt           = 49, 
    cudaDevAttrPciDomainId                    = 50, 
    cudaDevAttrTexturePitchAlignment          = 51, 
    cudaDevAttrMaxTextureCubemapWidth         = 52, 
    cudaDevAttrMaxTextureCubemapLayeredWidth  = 53, 
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54, 
    cudaDevAttrMaxSurface1DWidth              = 55, 
    cudaDevAttrMaxSurface2DWidth              = 56, 
    cudaDevAttrMaxSurface2DHeight             = 57, 
    cudaDevAttrMaxSurface3DWidth              = 58, 
    cudaDevAttrMaxSurface3DHeight             = 59, 
    cudaDevAttrMaxSurface3DDepth              = 60, 
    cudaDevAttrMaxSurface1DLayeredWidth       = 61, 
    cudaDevAttrMaxSurface1DLayeredLayers      = 62, 
    cudaDevAttrMaxSurface2DLayeredWidth       = 63, 
    cudaDevAttrMaxSurface2DLayeredHeight      = 64, 
    cudaDevAttrMaxSurface2DLayeredLayers      = 65, 
    cudaDevAttrMaxSurfaceCubemapWidth         = 66, 
    cudaDevAttrMaxSurfaceCubemapLayeredWidth  = 67, 
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68, 
    cudaDevAttrMaxTexture1DLinearWidth        = 69, 
    cudaDevAttrMaxTexture2DLinearWidth        = 70, 
    cudaDevAttrMaxTexture2DLinearHeight       = 71, 
    cudaDevAttrMaxTexture2DLinearPitch        = 72, 
    cudaDevAttrMaxTexture2DMipmappedWidth     = 73, 
    cudaDevAttrMaxTexture2DMipmappedHeight    = 74, 
    cudaDevAttrComputeCapabilityMajor         = 75,  
    cudaDevAttrComputeCapabilityMinor         = 76, 
    cudaDevAttrMaxTexture1DMipmappedWidth     = 77, 
    cudaDevAttrStreamPrioritiesSupported      = 78, 
    cudaDevAttrGlobalL1CacheSupported         = 79, 
    cudaDevAttrLocalL1CacheSupported          = 80, 
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81, 
    cudaDevAttrMaxRegistersPerMultiprocessor  = 82, 
    cudaDevAttrManagedMemory                  = 83, 
    cudaDevAttrIsMultiGpuBoard                = 84, 
    cudaDevAttrMultiGpuBoardGroupID           = 85, 
    cudaDevAttrHostNativeAtomicSupported      = 86, 
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, 
    cudaDevAttrPageableMemoryAccess           = 88, 
    cudaDevAttrConcurrentManagedAccess        = 89, 
    cudaDevAttrComputePreemptionSupported     = 90, 
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91 
};





enum  cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank              = 1, 
    cudaDevP2PAttrAccessSupported              = 2, 
    cudaDevP2PAttrNativeAtomicSupported        = 3  
};



struct  cudaDeviceProp
{
    char   name[256];                  
    size_t totalGlobalMem;             
    size_t sharedMemPerBlock;          
    int    regsPerBlock;               
    int    warpSize;                   
    size_t memPitch;                   
    int    maxThreadsPerBlock;         
    int    maxThreadsDim[3];           
    int    maxGridSize[3];             
    int    clockRate;                  
    size_t totalConstMem;              
    int    major;                      
    int    minor;                      
    size_t textureAlignment;           
    size_t texturePitchAlignment;      
    int    deviceOverlap;              
    int    multiProcessorCount;        
    int    kernelExecTimeoutEnabled;   
    int    integrated;                 
    int    canMapHostMemory;           
    int    computeMode;                
    int    maxTexture1D;               
    int    maxTexture1DMipmap;         
    int    maxTexture1DLinear;         
    int    maxTexture2D[2];            
    int    maxTexture2DMipmap[2];      
    int    maxTexture2DLinear[3];      
    int    maxTexture2DGather[2];      
    int    maxTexture3D[3];            
    int    maxTexture3DAlt[3];         
    int    maxTextureCubemap;          
    int    maxTexture1DLayered[2];     
    int    maxTexture2DLayered[3];     
    int    maxTextureCubemapLayered[2];
    int    maxSurface1D;               
    int    maxSurface2D[2];            
    int    maxSurface3D[3];            
    int    maxSurface1DLayered[2];     
    int    maxSurface2DLayered[3];     
    int    maxSurfaceCubemap;          
    int    maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;           
    int    concurrentKernels;          
    int    ECCEnabled;                 
    int    pciBusID;                   
    int    pciDeviceID;                
    int    pciDomainID;                
    int    tccDriver;                  
    int    asyncEngineCount;           
    int    unifiedAddressing;          
    int    memoryClockRate;            
    int    memoryBusWidth;             
    int    l2CacheSize;                
    int    maxThreadsPerMultiProcessor;
    int    streamPrioritiesSupported;  
    int    globalL1CacheSupported;     
    int    localL1CacheSupported;      
    size_t sharedMemPerMultiprocessor; 
    int    regsPerMultiprocessor;      
    int    managedMemory;              
    int    isMultiGpuBoard;            
    int    multiGpuBoardGroupID;       
    int    hostNativeAtomicSupported;  
    int    singleToDoublePrecisionPerfRatio; 
    int    pageableMemoryAccess;       
    int    concurrentManagedAccess;    
};















































































typedef  struct  cudaIpcEventHandle_st
{
    char reserved[64];
}cudaIpcEventHandle_t;




typedef  struct  cudaIpcMemHandle_st 
{
    char reserved[64];
}cudaIpcMemHandle_t;










typedef  enum cudaError cudaError_t;




typedef  struct CUstream_st *cudaStream_t;




typedef  struct CUevent_st *cudaEvent_t;




typedef  struct cudaGraphicsResource *cudaGraphicsResource_t;




typedef  struct CUuuid_st cudaUUID_t;




typedef  enum cudaOutputMode cudaOutputMode_t;


 

#line 1509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"

#line 60 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\builtin_types.h"

#line 62 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"


























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"



































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"
























enum  cudaSurfaceBoundaryMode
{
    cudaBoundaryModeZero  = 0,    
    cudaBoundaryModeClamp = 1,    
    cudaBoundaryModeTrap  = 2     
};




enum   cudaSurfaceFormatMode
{
    cudaFormatModeForced = 0,     
    cudaFormatModeAuto = 1        
};




struct  surfaceReference
{
    


    struct cudaChannelFormatDesc channelDesc;
};




typedef  unsigned long long cudaSurfaceObject_t;


 

#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"
#line 63 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"


























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"



































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
























enum  cudaTextureAddressMode
{
    cudaAddressModeWrap   = 0,    
    cudaAddressModeClamp  = 1,    
    cudaAddressModeMirror = 2,    
    cudaAddressModeBorder = 3     
};




enum  cudaTextureFilterMode
{
    cudaFilterModePoint  = 0,     
    cudaFilterModeLinear = 1      
};




enum  cudaTextureReadMode
{
    cudaReadModeElementType     = 0,  
    cudaReadModeNormalizedFloat = 1   
};




struct  textureReference
{
    


    int                          normalized;
    


    enum cudaTextureFilterMode   filterMode;
    


    enum cudaTextureAddressMode  addressMode[3];
    


    struct cudaChannelFormatDesc channelDesc;
    


    int                          sRGB;
    


    unsigned int                 maxAnisotropy;
    


    enum cudaTextureFilterMode   mipmapFilterMode;
    


    float                        mipmapLevelBias;
    


    float                        minMipmapLevelClamp;
    


    float                        maxMipmapLevelClamp;
    int                          __cudaReserved[15];
};




struct  cudaTextureDesc
{
    


    enum cudaTextureAddressMode addressMode[3];
    


    enum cudaTextureFilterMode  filterMode;
    


    enum cudaTextureReadMode    readMode;
    


    int                         sRGB;
    


    float                       borderColor[4];
    


    int                         normalizedCoords;
    


    unsigned int                maxAnisotropy;
    


    enum cudaTextureFilterMode  mipmapFilterMode;
    


    float                       mipmapLevelBias;
    


    float                       minMipmapLevelClamp;
    


    float                       maxMipmapLevelClamp;
};




typedef  unsigned long long cudaTextureObject_t;


 

#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#line 64 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"




























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_types.h"




































































#line 70 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_types.h"
#line 57 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"


#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"



































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1509 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\driver_types.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"

#line 62 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"






















































































































#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_types.h"
#line 63 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
























































































































































































































#line 218 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_types.h"
#line 64 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"














































































































































































































































































































































































































































#line 432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#line 65 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"
#line 62 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"

#line 64 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
















































































































































































































































#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
#line 65 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"






















#line 89 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"







#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"

struct  char1
{
    signed char x;
};

struct  uchar1
{
    unsigned char x;
};


struct  __attribute__((aligned(2))) char2
{
    signed char x, y;
};

struct  __attribute__((aligned(2))) uchar2
{
    unsigned char x, y;
};

struct  char3
{
    signed char x, y, z;
};

struct  uchar3
{
    unsigned char x, y, z;
};

struct  __attribute__((aligned(4))) char4
{
    signed char x, y, z, w;
};

struct  __attribute__((aligned(4))) uchar4
{
    unsigned char x, y, z, w;
};

struct  short1
{
    short x;
};

struct  ushort1
{
    unsigned short x;
};

struct  __attribute__((aligned(4))) short2
{
    short x, y;
};

struct  __attribute__((aligned(4))) ushort2
{
    unsigned short x, y;
};

struct  short3
{
    short x, y, z;
};

struct  ushort3
{
    unsigned short x, y, z;
};

struct  __attribute__((aligned(8))) short4 { short x; short y; short z; short w; };
struct  __attribute__((aligned(8))) ushort4 { unsigned short x; unsigned short y; unsigned short z; unsigned short w; };

struct  int1
{
    int x;
};

struct  uint1
{
    unsigned int x;
};

struct  __attribute__((aligned(8))) int2 { int x; int y; };
struct  __attribute__((aligned(8))) uint2 { unsigned int x; unsigned int y; };

struct  int3
{
    int x, y, z;
};

struct  uint3
{
    unsigned int x, y, z;
};

struct  __attribute__((aligned(16))) int4
{
    int x, y, z, w;
};

struct  __attribute__((aligned(16))) uint4
{
    unsigned int x, y, z, w;
};

struct  long1
{
    long int x;
};

struct  ulong1
{
    unsigned long x;
};


struct  __attribute__((aligned(8))) long2 { long int x; long int y; };
struct  __attribute__((aligned(8))) ulong2 { unsigned long int x; unsigned long int y; };












#line 231 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"

struct  long3
{
    long int x, y, z;
};

struct  ulong3
{
    unsigned long int x, y, z;
};

struct  __attribute__((aligned(16))) long4
{
    long int x, y, z, w;
};

struct  __attribute__((aligned(16))) ulong4
{
    unsigned long int x, y, z, w;
};

struct  float1
{
    float x;
};















#line 273 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"

struct  __attribute__((aligned(8))) float2 { float x; float y; };

#line 277 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"


struct  float3
{
    float x, y, z;
};

struct  __attribute__((aligned(16))) float4
{
    float x, y, z, w;
};

struct  longlong1
{
    long long int x;
};

struct  ulonglong1
{
    unsigned long long int x;
};

struct  __attribute__((aligned(16))) longlong2
{
    long long int x, y;
};

struct  __attribute__((aligned(16))) ulonglong2
{
    unsigned long long int x, y;
};

struct  longlong3
{
    long long int x, y, z;
};

struct  ulonglong3
{
    unsigned long long int x, y, z;
};

struct  __attribute__((aligned(16))) longlong4
{
    long long int x, y, z ,w;
};

struct  __attribute__((aligned(16))) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct  double1
{
    double x;
};

struct  __attribute__((aligned(16))) double2
{
    double x, y;
};

struct  double3
{
    double x, y, z;
};

struct  __attribute__((aligned(16))) double4
{
    double x, y, z, w;
};





#line 355 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"







typedef  struct char1 char1;
typedef  struct uchar1 uchar1;
typedef  struct char2 char2;
typedef  struct uchar2 uchar2;
typedef  struct char3 char3;
typedef  struct uchar3 uchar3;
typedef  struct char4 char4;
typedef  struct uchar4 uchar4;
typedef  struct short1 short1;
typedef  struct ushort1 ushort1;
typedef  struct short2 short2;
typedef  struct ushort2 ushort2;
typedef  struct short3 short3;
typedef  struct ushort3 ushort3;
typedef  struct short4 short4;
typedef  struct ushort4 ushort4;
typedef  struct int1 int1;
typedef  struct uint1 uint1;
typedef  struct int2 int2;
typedef  struct uint2 uint2;
typedef  struct int3 int3;
typedef  struct uint3 uint3;
typedef  struct int4 int4;
typedef  struct uint4 uint4;
typedef  struct long1 long1;
typedef  struct ulong1 ulong1;
typedef  struct long2 long2;
typedef  struct ulong2 ulong2;
typedef  struct long3 long3;
typedef  struct ulong3 ulong3;
typedef  struct long4 long4;
typedef  struct ulong4 ulong4;
typedef  struct float1 float1;
typedef  struct float2 float2;
typedef  struct float3 float3;
typedef  struct float4 float4;
typedef  struct longlong1 longlong1;
typedef  struct ulonglong1 ulonglong1;
typedef  struct longlong2 longlong2;
typedef  struct ulonglong2 ulonglong2;
typedef  struct longlong3 longlong3;
typedef  struct ulonglong3 ulonglong3;
typedef  struct longlong4 longlong4;
typedef  struct ulonglong4 ulonglong4;
typedef  struct double1 double1;
typedef  struct double2 double2;
typedef  struct double3 double3;
typedef  struct double4 double4;







struct  dim3
{
    unsigned int x, y, z;




#line 425 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
};

typedef  struct dim3 dim3;



#line 432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#line 65 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\builtin_types.h"
#line 285 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"
#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"




















































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"














































































































































































































































































































































































































































#line 432 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\vector_types.h"
#line 54 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"






#line 61 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"


#line 64 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"

#line 66 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"



#line 70 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"

uint3  extern const threadIdx;
uint3  extern const blockIdx;
dim3  extern const blockDim;
dim3  extern const gridDim;
int  extern const warpSize;





#line 82 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"






#line 89 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"






#line 96 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"






#line 103 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"






#line 110 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"






#line 117 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"

#line 119 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\device_launch_parameters.h"
#line 286 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"










































#line 44 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"






#line 51 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 55 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 63 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 67 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 71 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 75 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 79 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 87 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 91 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 99 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 115 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 127 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 135 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 139 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 143 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"



#line 147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"

#line 149 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\crt\\storage_class.h"
#line 287 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\crt/device_runtime.h"
#line 196 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\vcruntime.h"
#line 66 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
struct HMat {
#line 67 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
double *e;
#line 68 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
double *f;
#line 69 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short N;
#line 71 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *Ks;
#line 72 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float *Cms;
#line 73 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short NModels;
#line 74 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short NComps;
#line 75 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *boolModel;
#line 77 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short Depth;
#line 78 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short LognDepth;
#line 79 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short nFathers;
#line 80 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short nCallForFather;
#line 81 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *Fathers;
#line 83 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *SonNoVec;
#line 85 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *RelStarts;
#line 86 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *RelEnds;
#line 87 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *RelVec;
#line 88 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *SegStartI;
#line 89 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *SegEndI;
#line 90 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *SegToComp;
#line 91 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *MidComps;
#line 93 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *FIdxs;
#line 96 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *CompByLevel32;
#line 97 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *CompByFLevel32;
#line 98 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short nLRel;
#line 99 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *LRelStarts;
#line 100 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *LRelEnds;
#line 101 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short nFLRel;
#line 102 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *FLRelStarts;
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *FLRelEnds;};
#line 108 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
typedef struct HMat HMat;
#line 111 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
struct Stim {
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short NStimuli;
#line 113 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short loc;
#line 114 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short comp;
#line 115 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short numofdts;
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float area;
#line 117 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *dtInds;
#line 118 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float *amps;
#line 119 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float *durs;
#line 120 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float *dels;
#line 121 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float Nt;
#line 121 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
char __nv_no_debug_dummy_end_padding_0[4];};
#line 122 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
typedef struct Stim Stim;
#line 126 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
struct Sim {
#line 127 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float *Vs;
#line 128 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float dt;
#line 129 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float TFinal;
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
float Celsius;
#line 131 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short NRecSites;
#line 132 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
unsigned short *RecSites;};
#line 133 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
typedef struct Sim Sim;



#line 138 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"



#line 142 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"



#line 146 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"



#line 150 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"



#line 154 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"



#line 158 "c:\\pyneurogpu_win\\neurogpu6\\Util.h"
#line 106 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
extern  __inline__ float _Z3expf(float);
#line 121 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
extern  __inline__ float _Z4fabsf(float);
#line 248 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
extern  __inline__ float _Z3powff(float, float);
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static  __inline__ void _ZN39_INTERNAL_17_CudaStuff_cpp1_ii_1abe6ff811syncthreadsEv(void);
#line 82 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  float _Z9Cuefun_caf(float);
#line 89 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  float _Z9Cuefun_kmf(float);
#line 96 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  float _Z9Cuefun_kvf(float);
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  float _Z10Cutrap0_naffff(float, float, float, float);
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z11Cutrates_cafffRfS_S_S_(float, float, float, float *, float *, float *, float *);
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z10Curates_cafffRfS_S_S_(float, float, float, float *, float *, float *, float *);
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z11Curates_kcafffffRfS_S_S_(float, float, float, float, float, float *, float *, float *, float *);
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z11Cutrates_kmffffffRfS_S_S_(float, float, float, float, float, float, float *, float *, float *, float *);
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z10Curates_kmffffffRfS_S_S_(float, float, float, float, float, float, float *, float *, float *, float *);
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z11Cutrates_kvffffffRfS_S_S_(float, float, float, float, float, float, float *, float *, float *, float *);
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z10Curates_kvffffffRfS_S_S_(float, float, float, float, float, float, float *, float *, float *, float *);
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z11Cutrates_nafffffffffffffRfS_S_S_(float, float, float, float, float, float, float, float, float, float, float, float, float, float *, float *, float *, float *);
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z10Curates_nafffffffffffffRfS_S_S_(float, float, float, float, float, float, float, float, float, float, float, float, float, float *, float *, float *, float *);
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z14CuInitModel_cafRfS_fffS_(float, float *, float *, float, float, float, float *);
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z15CuInitModel_cadfRffS_(float, float *, float, float *);
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z15CuInitModel_kcafRffffff(float, float *, float, float, float, float, float);
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z14CuInitModel_kmfRffffff(float, float *, float, float, float, float, float);
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z14CuInitModel_kvfRffffff(float, float *, float, float, float, float, float);
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z14CuInitModel_nafRfS_ffffffffffff(float, float *, float *, float, float, float, float, float, float, float, float, float, float, float, float);
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z15CuDerivModel_caffRfS_fffS_(float, float, float *, float *, float, float, float, float *);
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z16CuDerivModel_cadffRffS_(float, float, float *, float, float *);
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z16CuDerivModel_kcaffRffffff(float, float, float *, float, float, float, float, float);
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z15CuDerivModel_kmffRffffff(float, float, float *, float, float, float, float, float);
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z15CuDerivModel_kvffRffffff(float, float, float *, float, float, float, float, float);
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z15CuDerivModel_naffRfS_ffffffffffff(float, float, float *, float *, float, float, float, float, float, float, float, float, float, float, float, float);
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z20CuBreakpointModel_caRdRffS0_S0_fffS0_(double *, float *, float, float *, float *, float, float, float, float *);
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z21CuBreakpointModel_cadRdRffS0_fS0_(double *, float *, float, float *, float, float *);
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z21CuBreakpointModel_kcaRdRffS0_fffff(double *, float *, float, float *, float, float, float, float, float);
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z20CuBreakpointModel_kmRdRffS0_fffff(double *, float *, float, float *, float, float, float, float, float);
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z20CuBreakpointModel_kvRdRffS0_fffff(double *, float *, float, float *, float, float, float, float, float);
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
extern  void _Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff(double *, float *, float, float *, float *, float, float, float, float, float, float, float, float, float, float, float, float);
#line 179 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
extern  void _Z8BeforeLU4HMatPdS0_t(HMat, double *, double *, unsigned short);
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
extern  void _Z5BkSub4HMatPdS0_S0_S0_t(HMat, double *, double *, double *, double *, unsigned short);
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
extern  void _Z13runSimulation4HMatPfS0_4Stim3SimS0_S0_S0_t(HMat, float *, float *, Stim, Sim, float *, float *, float *, unsigned short);
#line 476 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__attribute__((global)) __attribute__((__used__)) extern void _Z14NeuroGPUKernel4StimPf3Sim4HMatS0_S0_tt(Stim, float *, Sim, HMat, float *, float *, unsigned short, unsigned short);
#line 9 "c:\\pyneurogpu_win\\neurogpu6\\CudaStuff.cuh"
extern  __attribute__((shared)) char smem[];
#line 5 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) float cCm[384];
#line 6 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) double cE[384];
#line 7 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) double cF[384];
#line 8 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cFIdxs[2304];
#line 9 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cKs[384];
#line 10 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cSegToComp[384];
#line 11 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cBoolModel[2304];
#line 12 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cRelStarts[99];
#line 13 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cRelEnds[99];
#line 14 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cFathers[99];
#line 15 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cRelVec[186];
#line 16 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cSegStartI[187];
#line 17 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cSegEndI[187];
#line 19 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cCompByLevel32[896];
#line 20 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cCompByFLevel32[896];
#line 21 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cLRelStarts[24];
#line 22 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cLRelEnds[24];
#line 23 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cFLRelStarts[23];
#line 24 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cFLRelEnds[23];
#line 25 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 __attribute__((constant))  __attribute__((__used__)) unsigned short cSonNoVec[384];
#line 1 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\common_functions.h"












































































































































































































































#line 238 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\common_functions.h"








#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_device_runtime_api.h"





























































struct cudaFuncAttributes;

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaMalloc(void **p, size_t s) 
{ 
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *p, const void *c) 
{ 
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaGetDevice(int *device)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize)
{
  return cudaErrorUnknown;
}

__attribute__((device)) __attribute__((nv_weak)) cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *numBlocks, const void *func, int blockSize, size_t dynamicSmemSize, unsigned int flags)
{
  return cudaErrorUnknown;
}

#line 95 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_device_runtime_api.h"



































































































































#line 227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_device_runtime_api.h"

#line 229 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\cuda_device_runtime_api.h"
#line 247 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\common_functions.h"
#line 248 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\common_functions.h"

#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"





































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 9831 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"





#line 9837 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"




#line 65 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"



#line 69 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"












 float __nv_fast_sinf(float x) ;












 float __nv_fast_cosf(float x) ;


























 float __nv_fast_log2f(float x) ;














 float __nv_fast_tanf(float x) ;















 void __nv_fast_sincosf(float x, float *sptr, float *cptr) ; 


















































 float __nv_fast_expf(float x) ;
































 float __nv_fast_exp10f(float x) ;




























 float __nv_fast_log10f(float x) ;












































 float __nv_fast_logf(float x) ;











































 float __nv_fast_powf(float x, float y) ;












 int __nv_hadd(int x, int y) ;













 int __nv_rhadd(int x, int y) ;












 unsigned int __nv_uhadd(unsigned int x, unsigned int y) ;













 unsigned int __nv_urhadd(unsigned int x, unsigned int y) ;












 float __nv_fsub_rn (float x, float y) ;












 float __nv_fsub_rz (float x, float y) ;












 float __nv_fsub_rd (float x, float y) ;












 float __nv_fsub_ru (float x, float y) ;







































 float __nv_frsqrt_rn (float x) ;











 int __nv_ffs(int x) ;











 int __nv_ffsll(long long int x) ;











 float __nv_rintf(float x) ;












 long long int __nv_llrintf(float x) ;

















































 float __nv_nearbyintf(float x) ;












 int __nv_signbitf(float x) ;









 float __nv_copysignf(float x, float y) ;










 int __nv_finitef(float x) ;












 int __nv_isinff(float x) ;











 int __nv_isnanf(float x) ;
































 float __nv_nextafterf(float x, float y) ;












 float __nv_nanf(const signed char *tagp) ;









































 float __nv_sinf(float x) ;

































 float __nv_cosf(float x) ;















 void __nv_sincosf(float x, float *sptr, float *cptr) ;




























































 float __nv_sinpif(float x) ;




















































 float __nv_cospif(float x) ;






























 void __nv_sincospif(float x, float *sptr, float *cptr) ;









































 float __nv_tanf(float x) ;



















































 float __nv_log2f(float x) ;







































 float __nv_expf(float x) ;





















 float __nv_exp10f(float x) ;































 float __nv_coshf(float x) ;






























 float __nv_sinhf(float x) ;






























 float __nv_tanhf(float x) ;

































 float __nv_atan2f(float x, float y) ;































 float __nv_atanf(float x) ;
































 float __nv_asinf(float x) ;























 float __nv_acosf(float x) ;







































































 float __nv_logf(float x) ;



















































 float __nv_log10f(float x) ;





























































































 float __nv_log1pf(float x) ;


































 float __nv_acoshf(float x) ;












 float __nv_asinhf(float x) ;


















































 float __nv_atanhf(float x) ;









































 float __nv_expm1f(float x) ;








































 float __nv_hypotf(float x, float y) ;













































 float __nv_rhypotf(float x, float y) ;



















































 float __nv_rnormf(int dim, float const * a) ;













































 float __nv_normf(int dim, float const * a) ;













































 float __nv_norm3df(float a, float b, float c) ;

















































 float __nv_rnorm3df(float a, float b, float c) ;

















































 float __nv_norm4df(float a, float b, float c, float d) ;






















































 float __nv_rnorm4df(float a, float b, float c, float d) ;


















































































 float __nv_cbrtf(float x) ;

















































 float __nv_rcbrtf(float x) ;






































 float __nv_j0f(float x) ;

























































 float __nv_j1f(float x) ;
















































 float __nv_y0f(float x) ;
















































 float __nv_y1f(float x) ;

















































 float __nv_ynf(int n, float x) ;







































 float __nv_jnf(int n, float x) ;


























 float __nv_cyl_bessel_i0f(float x) ;


























 float __nv_cyl_bessel_i1f(float x) ;














































































 float __nv_erff(float x) ;

























































 float __nv_erfinvf(float x) ;


































 float __nv_erfcf(float x) ;











































































 float __nv_erfcxf(float x) ;
























































 float __nv_erfcinvf(float x) ;


























































 float __nv_normcdfinvf(float x) ;











































 float __nv_normcdff(float x) ;




























































































































 float __nv_lgammaf(float x) ;
























































 float __nv_ldexpf(float x, int y) ;








































































 float __nv_scalbnf(float x, int y) ;











































































 float __nv_frexpf(float x, int *b) ;
























































 float __nv_modff(float x, float *b) ;



























































 float __nv_fmodf(float x, float y) ;





















































































 float __nv_remainderf(float x, float y) ;


















































 float __nv_remquof(float x, float y, int* quo) ;


























































































































































 float __nv_fmaf(float x, float y, float z) ;




















































































































































































































































































































 float __nv_powif(float x, int y) ;




















































































































































































































































































































 double __nv_powi(double x, int y) ;




















































































































































































































































































































 float __nv_powf(float x, float y) ;









































































































 float __nv_tgammaf(float x) ;













 float __nv_roundf(float x) ;














 long long int __nv_llroundf(float x) ;






















 float __nv_fdimf(float x, float y) ;


























 int __nv_ilogbf(float x) ;



















































 float __nv_logbf(float x) ;











 double __nv_rint(double x) ;












 long long int __nv_llrint(double x) ;

















































 double __nv_nearbyint(double x) ;












 int __nv_signbitd(double x) ;












 int __nv_isfinited(double x) ;












 int __nv_isinfd(double x) ;











 int __nv_isnand(double x) ;









 double __nv_copysign(double x, double y) ;















 void __nv_sincos(double x, double *sptr, double *cptr) ;






























 void __nv_sincospi(double x, double *sptr, double *cptr) ;









































 double __nv_sin(double x) ;

































 double __nv_cos(double x) ;




























































 double __nv_sinpi(double x) ;




















































 double __nv_cospi(double x) ;









































 double __nv_tan(double x) ;







































































 double __nv_log(double x) ;



















































 double __nv_log2(double x) ;



















































 double __nv_log10(double x) ;





























































































 double __nv_log1p(double x) ;







































 double __nv_exp(double x) ;





















 double __nv_exp2(double x) ;





















 double __nv_exp10(double x) ;









































 double __nv_expm1(double x) ;































 double __nv_cosh(double x) ;






























 double __nv_sinh(double x) ;






























 double __nv_tanh(double x) ;

































 double __nv_atan2(double x, double y) ;































 double __nv_atan(double x) ;
































 double __nv_asin(double x) ;























 double __nv_acos(double x) ;


































 double __nv_acosh(double x) ;












 double __nv_asinh(double x) ;


















































 double __nv_atanh(double x) ;








































 double __nv_hypot(double x, double y) ;











































 double __nv_rhypot(double x, double y) ;










































 double __nv_norm3d(double a, double b, double c) ;

















































 double __nv_rnorm3d(double a, double b, double c) ;
















































 double __nv_norm4d(double a, double b, double c, double d) ;













































 double __nv_norm(int dim, double const * a) ;



















































 double __nv_rnorm(int dim, double const * a) ;


















































































 double __nv_cbrt(double x) ;






















































 double __nv_rnorm4d(double a, double b, double c, double d) ;

















































 double __nv_rcbrt(double x) ;




















































































































































































































































































































 double __nv_pow(double x, double y) ;






































 double __nv_j0(double x) ;

























































 double __nv_j1(double x) ;
















































 double __nv_y0(double x) ;
















































 double __nv_y1(double x) ;

















































 double __nv_yn(int n, double x) ;







































 double __nv_jn(int n, double x) ;


























 double __nv_cyl_bessel_i0(double x) ;


























 double __nv_cyl_bessel_i1(double x) ;














































































 double __nv_erf(double x) ;

























































 double __nv_erfinv(double x) ;
























































 double __nv_erfcinv(double x) ;


























































 double __nv_normcdfinv(double x) ;


































 double __nv_erfc(double x) ;











































































 double __nv_erfcx(double x) ;











































 double __nv_normcdf(double x) ;









































































































 double __nv_tgamma(double x) ;




























































































































 double __nv_lgamma(double x) ;
























































 double __nv_ldexp(double x, int y) ;








































































 double __nv_scalbn(double x, int y) ;











































































 double __nv_frexp(double x, int *b) ;
























































 double __nv_modf(double x, double *b) ;



























































 double __nv_fmod(double x, double y) ;





















































































 double __nv_remainder(double x, double y) ;


















































 double __nv_remquo(double x, double y, int *c) ;
































 double __nv_nextafter(double x, double y) ;












 double __nv_nan(const signed char *tagp) ;













 double __nv_round(double x) ;














 long long int __nv_llround(double x) ;






















 double __nv_fdim(double x, double y) ;


























 int __nv_ilogb(double x) ;



















































 double __nv_logb(double x) ;


























































































































































 double __nv_fma(double x, double y, double z) ;









 int __nv_clz(int x) ;








 int __nv_clzll(long long x) ;









 int __nv_popc(int x) ;








 int __nv_popcll(long long x) ;
























 unsigned int __nv_byte_perm(unsigned int x, unsigned int y, unsigned int z) ;










 int __nv_min(int x, int y) ;









 unsigned int __nv_umin(unsigned int x, unsigned int y) ;









 long long __nv_llmin(long long x, long long y) ;









 unsigned long long __nv_ullmin(unsigned long long x, unsigned long long y) ;
    









 int __nv_max(int x, int y) ;









 unsigned int __nv_umax(unsigned int x, unsigned int y) ;









 long long __nv_llmax(long long x, long long y) ;









 unsigned long long __nv_ullmax(unsigned long long x, unsigned long long y) ;










 int __nv_mulhi(int x, int y) ;









 unsigned int __nv_umulhi(unsigned int x, unsigned int y) ;









 long long __nv_mul64hi(long long x, long long y) ;









 unsigned long long __nv_umul64hi(unsigned long long x, unsigned long long y) ;










 int __nv_mul24(int x, int y) ;









 unsigned int __nv_umul24(unsigned int x, unsigned int y) ;









 unsigned int __nv_brev(unsigned int x) ;
    








 unsigned long long __nv_brevll(unsigned long long x) ;




































































 int __nv_sad(int x, int y, int z) ;




































































 unsigned int __nv_usad(unsigned int x, unsigned int y, unsigned int z) ;









 int __nv_abs(int x) ;










 long long __nv_llabs(long long x) ;



















































 float __nv_floorf(float f) ;



















































 double __nv_floor(double f) ;









































 float __nv_fabsf(float f) ;









































 double __nv_fabs(double f) ;


 double __nv_rcp64h(double d) ;
















 float __nv_fminf(float x, float y) ;
















 float __nv_fmaxf(float x, float y) ;





































































 float __nv_rsqrtf(float x) ;
















 double __nv_fmin(double x, double y) ;
















 double __nv_fmax(double x, double y) ;





































































 double __nv_rsqrt(double x) ;



























































 double __nv_ceil(double x) ;











 double __nv_trunc(double x) ;





















 float __nv_exp2f(float x) ;











 float __nv_truncf(float x) ;



























































 float __nv_ceilf(float x) ;























 float __nv_saturatef(float x) ;

























































































































































 float __nv_fmaf_rn(float x, float y, float z) ;
























































































































































 float __nv_fmaf_rz(float x, float y, float z) ;
























































































































































 float __nv_fmaf_rd(float x, float y, float z) ;
























































































































































 float __nv_fmaf_ru(float x, float y, float z) ;


 float __nv_fmaf_ieee_rn(float x, float y, float z) ;

 float __nv_fmaf_ieee_rz(float x, float y, float z) ;

 float __nv_fmaf_ieee_rd(float x, float y, float z) ;

 float __nv_fmaf_ieee_ru(float x, float y, float z) ;





























































































































































 double __nv_fma_rn(double x, double y, double z) ;




























































































































































 double __nv_fma_rz(double x, double y, double z) ;




























































































































































 double __nv_fma_rd(double x, double y, double z) ;




























































































































































 double __nv_fma_ru(double x, double y, double z) ;











































































 float __nv_fast_fdividef(float x, float y) ;











 float __nv_fdiv_rn(float x, float y) ;










 float __nv_fdiv_rz(float x, float y) ;










 float __nv_fdiv_rd(float x, float y) ;










 float __nv_fdiv_ru(float x, float y) ;

































 float __nv_frcp_rn(float x) ;
































 float __nv_frcp_rz(float x) ;
































 float __nv_frcp_rd(float x) ;
































 float __nv_frcp_ru(float x) ;































 float __nv_fsqrt_rn(float x) ;






























 float __nv_fsqrt_rz(float x) ;






























 float __nv_fsqrt_rd(float x) ;






























 float __nv_fsqrt_ru(float x) ;












 double __nv_ddiv_rn(double x, double y) ;











 double __nv_ddiv_rz(double x, double y) ;











 double __nv_ddiv_rd(double x, double y) ;











 double __nv_ddiv_ru(double x, double y) ;


































 double __nv_drcp_rn(double x) ;

































 double __nv_drcp_rz(double x) ;

































 double __nv_drcp_rd(double x) ;

































 double __nv_drcp_ru(double x) ;
































 double __nv_dsqrt_rn(double x) ;
































 double __nv_dsqrt_rz(double x) ;































 double __nv_dsqrt_rd(double x) ;































 double __nv_dsqrt_ru(double x) ;





































































 float __nv_sqrtf(float x) ;





































































 double __nv_sqrt(double x) ;












 double __nv_dadd_rn(double x, double y) ;











 double __nv_dadd_rz(double x, double y) ;











 double __nv_dadd_rd(double x, double y) ;











 double __nv_dadd_ru(double x, double y) ;












 double __nv_dmul_rn(double x, double y) ;











 double __nv_dmul_rz(double x, double y) ;











 double __nv_dmul_rd(double x, double y) ;











 double __nv_dmul_ru(double x, double y) ;












 float __nv_fadd_rd(float x, float y) ;











 float __nv_fadd_ru(float x, float y) ;












 float __nv_fmul_rd(float x, float y) ;











 float __nv_fmul_ru(float x, float y) ;












 float __nv_fadd_rn(float x, float y) ;











 float __nv_fadd_rz(float x, float y) ;












 float __nv_fmul_rn(float x, float y) ;











 float __nv_fmul_rz(float x, float y) ;









 float __nv_double2float_rn(double d) ;








 float __nv_double2float_rz(double d) ;








 float __nv_double2float_rd(double d) ;








 float __nv_double2float_ru(double d) ;
    








 int __nv_double2int_rn(double d) ;








 int __nv_double2int_rz(double d) ;








 int __nv_double2int_rd(double d) ;








 int __nv_double2int_ru(double d) ;









 unsigned int __nv_double2uint_rn(double d) ;








 unsigned int __nv_double2uint_rz(double d) ;








 unsigned int __nv_double2uint_rd(double d) ;








 unsigned int __nv_double2uint_ru(double d) ;








 double __nv_int2double_rn(int i) ;








 double __nv_uint2double_rn(unsigned int i) ;









 int __nv_float2int_rn(float in) ;








 int __nv_float2int_rz(float in) ;








 int __nv_float2int_rd(float in) ;








 int __nv_float2int_ru(float in) ;








 unsigned int __nv_float2uint_rn(float in) ;








 unsigned int __nv_float2uint_rz(float in) ;








 unsigned int __nv_float2uint_rd(float in) ;








 unsigned int __nv_float2uint_ru(float in) ;









 float __nv_int2float_rn(int in) ;








 float __nv_int2float_rz(int in) ;








 float __nv_int2float_rd(int in) ;








 float __nv_int2float_ru(int in) ;









 float __nv_uint2float_rn(unsigned int in) ;








 float __nv_uint2float_rz(unsigned int in) ;








 float __nv_uint2float_rd(unsigned int in) ;








 float __nv_uint2float_ru(unsigned int in) ;










 double __nv_hiloint2double(int x, int y) ;








 int __nv_double2loint(double d) ;








 int __nv_double2hiint(double d) ;









 long long __nv_float2ll_rn(float f) ;








 long long __nv_float2ll_rz(float f) ;








 long long __nv_float2ll_rd(float f) ;








 long long __nv_float2ll_ru(float f) ;








 unsigned long long __nv_float2ull_rn(float f) ;








 unsigned long long __nv_float2ull_rz(float f) ;








 unsigned long long __nv_float2ull_rd(float f) ;








 unsigned long long __nv_float2ull_ru(float f) ;









 long long __nv_double2ll_rn(double f) ;








 long long __nv_double2ll_rz(double f) ;








 long long __nv_double2ll_rd(double f) ;








 long long __nv_double2ll_ru(double f) ;









 unsigned long long __nv_double2ull_rn(double f) ;








 unsigned long long __nv_double2ull_rz(double f) ;








 unsigned long long __nv_double2ull_rd(double f) ;








 unsigned long long __nv_double2ull_ru(double f) ;









 float __nv_ll2float_rn(long long l) ;








 float __nv_ll2float_rz(long long l) ;








 float __nv_ll2float_rd(long long l) ;








 float __nv_ll2float_ru(long long l) ;









 float __nv_ull2float_rn(unsigned long long l) ;








 float __nv_ull2float_rz(unsigned long long l) ;








 float __nv_ull2float_rd(unsigned long long l) ;








 float __nv_ull2float_ru(unsigned long long l) ;









 double __nv_ll2double_rn(long long l) ;








 double __nv_ll2double_rz(long long l) ;








 double __nv_ll2double_rd(long long l) ;








 double __nv_ll2double_ru(long long l) ;









 double __nv_ull2double_rn(unsigned long long l) ;








 double __nv_ull2double_rz(unsigned long long l) ;








 double __nv_ull2double_rd(unsigned long long l) ;








 double __nv_ull2double_ru(unsigned long long l) ;









 unsigned short __nv_float2half_rn(float f) ;








 float __nv_half2float(unsigned short h) ;








 float __nv_int_as_float(int x) ;









 int __nv_float_as_int(float x) ;
    








 float __nv_uint_as_float(unsigned int x) ;









 unsigned int __nv_float_as_uint(float x) ;
    








 double __nv_longlong_as_double(long long x) ;









 long long  __nv_double_as_longlong (double x) ;

































#line 12534 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"



#line 12538 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"



#line 12542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"

#line 9843 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 9845 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 3292 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"





#line 3298 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


#line 3301 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


#line 3304 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"




#line 3309 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"




#line 3314 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"






static __inline__ __attribute__((always_inline)) int __syncthreads_count(int predicate);

static __inline__ __attribute__((always_inline)) int __syncthreads_and(int predicate);

static __inline__ __attribute__((always_inline)) int __syncthreads_or(int predicate);






static __inline__ __attribute__((always_inline)) void __threadfence_block();

static __inline__ __attribute__((always_inline)) void __threadfence();

static __inline__ __attribute__((always_inline)) void __threadfence_system();






static __inline__ __attribute__((always_inline)) int __all(int a);

static __inline__ __attribute__((always_inline)) int __any(int a);

static __inline__ __attribute__((always_inline))


#line 3350 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
int
#line 3352 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
__ballot(int a);








#line 3362 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
static __inline__ __attribute__((always_inline)) void __brkpt();
#line 3364 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))


#line 3369 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
int
#line 3371 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
clock() ; 

static __inline__ __attribute__((always_inline)) long long clock64();
    


static __inline__ __attribute__((always_inline)) unsigned int __pm0(void);

static __inline__ __attribute__((always_inline)) unsigned int __pm1(void);

static __inline__ __attribute__((always_inline)) unsigned int __pm2(void);

static __inline__ __attribute__((always_inline)) unsigned int __pm3(void);

static __inline__ __attribute__((always_inline)) void __trap(void);

static __inline__ __attribute__((always_inline)) void* memcpy(void *dest, const void *src, size_t n) ;

static __inline__ __attribute__((always_inline)) void* memset(void *dest, int c, size_t n) ;






static __inline__ __attribute__((always_inline)) int __clz(int x);

static __inline__ __attribute__((always_inline)) int __clzll(long long x);



#line 3403 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
static __inline__ __attribute__((always_inline)) int __popc(int x);
#line 3405 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"



#line 3409 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
static __inline__ __attribute__((always_inline)) int __popcll(long long x);
#line 3411 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline)) unsigned int __byte_perm(unsigned int a,
                                                unsigned int b,
                                                unsigned int c);






static __inline__ __attribute__((always_inline)) int min(int x, int y);

static __inline__ __attribute__((always_inline)) unsigned int umin(unsigned int x, unsigned int y);
    
static __inline__ __attribute__((always_inline)) long long llmin(long long x, long long y);

static __inline__ __attribute__((always_inline)) unsigned long long ullmin(unsigned long long x,
                                                 unsigned long long y);
    
static __inline__ __attribute__((always_inline)) int max(int x, int y);

static __inline__ __attribute__((always_inline)) unsigned int umax(unsigned int x, unsigned int y);
    
static __inline__ __attribute__((always_inline)) long long llmax(long long x, long long y);

static __inline__ __attribute__((always_inline)) unsigned long long ullmax(unsigned long long x,
                                                 unsigned long long y);

static __inline__ __attribute__((always_inline)) int __mulhi(int x, int y);

static __inline__ __attribute__((always_inline)) unsigned int __umulhi(unsigned int x, unsigned int y);

static __inline__ __attribute__((always_inline)) long long __mul64hi(long long x, long long y);

static __inline__ __attribute__((always_inline)) unsigned long long __umul64hi(unsigned long long x,
                                                     unsigned long long y);

static __inline__ __attribute__((always_inline)) int __mul24(int x, int y);

static __inline__ __attribute__((always_inline)) unsigned int __umul24(unsigned int x, unsigned int y);

static __inline__ __attribute__((always_inline)) unsigned int __brev(unsigned int x);
    
static __inline__ __attribute__((always_inline)) unsigned long long __brevll(unsigned long long x);
    


#line 3459 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
static __inline__ __attribute__((always_inline)) int __sad(int x, int y, int z);
#line 3461 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline)) unsigned int __usad(unsigned int x,
                                           unsigned int y,
                                           unsigned int z);

static __inline__ __attribute__((always_inline)) int abs(int x) ;

static __inline__ __attribute__((always_inline)) long labs(long x) ;

static __inline__ __attribute__((always_inline)) long long llabs(long long x) ;






static __inline__ __attribute__((always_inline)) float floorf(float f);

static __inline__ __attribute__((always_inline)) double floor(double f);

static __inline__ __attribute__((always_inline)) float fabsf(float f);

static __inline__ __attribute__((always_inline)) double fabs(double f);

static __inline__ __attribute__((always_inline)) float fminf(float x, float y);

static __inline__ __attribute__((always_inline)) float fmaxf(float x, float y);

static __inline__ __attribute__((always_inline)) float rsqrtf(float x);

static __inline__ __attribute__((always_inline)) double fmin(double x, double y);

static __inline__ __attribute__((always_inline)) double fmax(double x, double y);

static __inline__ __attribute__((always_inline)) double rsqrt(double x);

static __inline__ __attribute__((always_inline)) double ceil(double x);

static __inline__ __attribute__((always_inline)) double trunc(double x);

static __inline__ __attribute__((always_inline)) float exp2f(float x);

static __inline__ __attribute__((always_inline)) float truncf(float x);

static __inline__ __attribute__((always_inline)) float ceilf(float x);

static __inline__ __attribute__((always_inline)) float __saturatef(float x);






static __inline__ __attribute__((always_inline)) float __fmaf_rn(float x, float y, float z);

static __inline__ __attribute__((always_inline)) float __fmaf_rz(float x, float y, float z);

static __inline__ __attribute__((always_inline)) float __fmaf_rd(float x, float y, float z);

static __inline__ __attribute__((always_inline)) float __fmaf_ru(float x, float y, float z);






static __inline__ __attribute__((always_inline)) float __fmaf_ieee_rn(float x, float y, float z);

static __inline__ __attribute__((always_inline)) float __fmaf_ieee_rz(float x, float y, float z);

static __inline__ __attribute__((always_inline)) float __fmaf_ieee_rd(float x, float y, float z);

static __inline__ __attribute__((always_inline)) float __fmaf_ieee_ru(float x, float y, float z);






static __inline__ __attribute__((always_inline)) double __fma_rn(double x, double y, double z);

static __inline__ __attribute__((always_inline)) double __fma_rz(double x, double y, double z);

static __inline__ __attribute__((always_inline)) double __fma_rd(double x, double y, double z);

static __inline__ __attribute__((always_inline)) double __fma_ru(double x, double y, double z);

static __inline__ __attribute__((always_inline)) float __fdividef(float x, float y);






static __inline__ __attribute__((always_inline)) float __fdiv_rn(float x, float y);

static __inline__ __attribute__((always_inline)) float __fdiv_rz(float x, float y);

static __inline__ __attribute__((always_inline)) float __fdiv_rd(float x, float y);

static __inline__ __attribute__((always_inline)) float __fdiv_ru(float x, float y);






static __inline__ __attribute__((always_inline)) float __frcp_rn(float x);

static __inline__ __attribute__((always_inline)) float __frcp_rz(float x);

static __inline__ __attribute__((always_inline)) float __frcp_rd(float x);

static __inline__ __attribute__((always_inline)) float __frcp_ru(float x);






static __inline__ __attribute__((always_inline)) float __fsqrt_rn(float x);

static __inline__ __attribute__((always_inline)) float __fsqrt_rz(float x);

static __inline__ __attribute__((always_inline)) float __fsqrt_rd(float x);

static __inline__ __attribute__((always_inline)) float __fsqrt_ru(float x);






static __inline__ __attribute__((always_inline)) double __ddiv_rn(double x, double y);

static __inline__ __attribute__((always_inline)) double __ddiv_rz(double x, double y);

static __inline__ __attribute__((always_inline)) double __ddiv_rd(double x, double y);

static __inline__ __attribute__((always_inline)) double __ddiv_ru(double x, double y);






static __inline__ __attribute__((always_inline)) double __drcp_rn(double x);

static __inline__ __attribute__((always_inline)) double __drcp_rz(double x);

static __inline__ __attribute__((always_inline)) double __drcp_rd(double x);

static __inline__ __attribute__((always_inline)) double __drcp_ru(double x);






static __inline__ __attribute__((always_inline)) double __dsqrt_rn(double x);

static __inline__ __attribute__((always_inline)) double __dsqrt_rz(double x);

static __inline__ __attribute__((always_inline)) double __dsqrt_rd(double x);

static __inline__ __attribute__((always_inline)) double __dsqrt_ru(double x);

static __inline__ __attribute__((always_inline)) float sqrtf(float x);

static __inline__ __attribute__((always_inline)) double sqrt(double x);






static __inline__ __attribute__((always_inline)) double __dadd_rn(double x, double y);

static __inline__ __attribute__((always_inline)) double __dadd_rz(double x, double y);

static __inline__ __attribute__((always_inline)) double __dadd_rd(double x, double y);

static __inline__ __attribute__((always_inline)) double __dadd_ru(double x, double y);






static __inline__ __attribute__((always_inline)) double __dmul_rn(double x, double y);

static __inline__ __attribute__((always_inline)) double __dmul_rz(double x, double y);

static __inline__ __attribute__((always_inline)) double __dmul_rd(double x, double y);

static __inline__ __attribute__((always_inline)) double __dmul_ru(double x, double y);






static __inline__ __attribute__((always_inline)) float __fadd_rd(float x, float y);

static __inline__ __attribute__((always_inline)) float __fadd_ru(float x, float y);

static __inline__ __attribute__((always_inline)) float __fadd_rn(float x, float y);

static __inline__ __attribute__((always_inline)) float __fadd_rz(float x, float y);






static __inline__ __attribute__((always_inline)) float __fmul_rd(float x, float y);

static __inline__ __attribute__((always_inline)) float __fmul_ru(float x, float y);

static __inline__ __attribute__((always_inline)) float __fmul_rn(float x, float y);

static __inline__ __attribute__((always_inline)) float __fmul_rz(float x, float y);







static __inline__ __attribute__((always_inline)) float __double2float_rn(double d);

static __inline__ __attribute__((always_inline)) float __double2float_rz(double d);

static __inline__ __attribute__((always_inline)) float __double2float_rd(double d);

static __inline__ __attribute__((always_inline)) float __double2float_ru(double d);
    

static __inline__ __attribute__((always_inline)) int __double2int_rn(double d);

static __inline__ __attribute__((always_inline)) int __double2int_rz(double d);

static __inline__ __attribute__((always_inline)) int __double2int_rd(double d);

static __inline__ __attribute__((always_inline)) int __double2int_ru(double d);


static __inline__ __attribute__((always_inline)) unsigned int __double2uint_rn(double d);

static __inline__ __attribute__((always_inline)) unsigned int __double2uint_rz(double d);

static __inline__ __attribute__((always_inline)) unsigned int __double2uint_rd(double d);

static __inline__ __attribute__((always_inline)) unsigned int __double2uint_ru(double d);


static __inline__ __attribute__((always_inline)) double __int2double_rn(int i);


static __inline__ __attribute__((always_inline)) double __uint2double_rn(unsigned int i);


static __inline__ __attribute__((always_inline)) int __float2int_rn(float in);

static __inline__ __attribute__((always_inline)) int __float2int_rz(float in);

static __inline__ __attribute__((always_inline)) int __float2int_rd(float in);

static __inline__ __attribute__((always_inline)) int __float2int_ru(float in);


static __inline__ __attribute__((always_inline)) unsigned int __float2uint_rn(float in);

static __inline__ __attribute__((always_inline)) unsigned int __float2uint_rz(float in);

static __inline__ __attribute__((always_inline)) unsigned int __float2uint_rd(float in);

static __inline__ __attribute__((always_inline)) unsigned int __float2uint_ru(float in);


static __inline__ __attribute__((always_inline)) float __int2float_rn(int in);

static __inline__ __attribute__((always_inline)) float __int2float_rz(int in);

static __inline__ __attribute__((always_inline)) float __int2float_rd(int in);

static __inline__ __attribute__((always_inline)) float __int2float_ru(int in);


static __inline__ __attribute__((always_inline)) float __uint2float_rn(unsigned int in);

static __inline__ __attribute__((always_inline)) float __uint2float_rz(unsigned int in);

static __inline__ __attribute__((always_inline)) float __uint2float_rd(unsigned int in);

static __inline__ __attribute__((always_inline)) float __uint2float_ru(unsigned int in);


static __inline__ __attribute__((always_inline)) double __hiloint2double(int a, int b);

static __inline__ __attribute__((always_inline)) int __double2loint(double d);

static __inline__ __attribute__((always_inline)) int __double2hiint(double d);


static __inline__ __attribute__((always_inline)) long long __float2ll_rn(float f);

static __inline__ __attribute__((always_inline)) long long __float2ll_rz(float f);

static __inline__ __attribute__((always_inline)) long long __float2ll_rd(float f);

static __inline__ __attribute__((always_inline)) long long __float2ll_ru(float f);


static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_rn(float f);

static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_rz(float f);

static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_rd(float f);

static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_ru(float f);


static __inline__ __attribute__((always_inline)) long long __double2ll_rn(double f);

static __inline__ __attribute__((always_inline)) long long __double2ll_rz(double f);

static __inline__ __attribute__((always_inline)) long long __double2ll_rd(double f);

static __inline__ __attribute__((always_inline)) long long __double2ll_ru(double f);


static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_rn(double f);

static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_rz(double f);

static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_rd(double f);

static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_ru(double f);


static __inline__ __attribute__((always_inline)) float __ll2float_rn(long long l);

static __inline__ __attribute__((always_inline)) float __ll2float_rz(long long l);

static __inline__ __attribute__((always_inline)) float __ll2float_rd(long long l);

static __inline__ __attribute__((always_inline)) float __ll2float_ru(long long l);


static __inline__ __attribute__((always_inline)) float __ull2float_rn(unsigned long long l);

static __inline__ __attribute__((always_inline)) float __ull2float_rz(unsigned long long l);

static __inline__ __attribute__((always_inline)) float __ull2float_rd(unsigned long long l);

static __inline__ __attribute__((always_inline)) float __ull2float_ru(unsigned long long l);


static __inline__ __attribute__((always_inline)) double __ll2double_rn(long long l);

static __inline__ __attribute__((always_inline)) double __ll2double_rz(long long l);

static __inline__ __attribute__((always_inline)) double __ll2double_rd(long long l);

static __inline__ __attribute__((always_inline)) double __ll2double_ru(long long l);


static __inline__ __attribute__((always_inline)) double __ull2double_rn(unsigned long long l);

static __inline__ __attribute__((always_inline)) double __ull2double_rz(unsigned long long l);

static __inline__ __attribute__((always_inline)) double __ull2double_rd(unsigned long long l);

static __inline__ __attribute__((always_inline)) double __ull2double_ru(unsigned long long l);

static __inline__ __attribute__((always_inline)) unsigned short __float2half_rn(float f);

static __inline__ __attribute__((always_inline)) float __half2float(unsigned short h);

static __inline__ __attribute__((always_inline)) float __int_as_float(int x);

static __inline__ __attribute__((always_inline)) int __float_as_int(float x);

static __inline__ __attribute__((always_inline)) float __uint_as_float(unsigned int x);

static __inline__ __attribute__((always_inline)) unsigned int __float_as_uint(float x);
    
static __inline__ __attribute__((always_inline)) double __longlong_as_double(long long x);

static __inline__ __attribute__((always_inline)) long long  __double_as_longlong (double x);







static __inline__ __attribute__((always_inline)) float __sinf(float a) ;

static __inline__ __attribute__((always_inline)) float __cosf(float a) ;

static __inline__ __attribute__((always_inline)) float __log2f(float a) ;







static __inline__ __attribute__((always_inline)) float __tanf(float a) ;

static __inline__ __attribute__((always_inline)) void __sincosf(float a, float *sptr, float *cptr) ;

static __inline__ __attribute__((always_inline)) float __expf(float a) ;

static __inline__ __attribute__((always_inline)) float __exp10f(float a) ;

static __inline__ __attribute__((always_inline)) float __log10f(float a) ;

static __inline__ __attribute__((always_inline)) float __logf(float a) ;

static __inline__ __attribute__((always_inline)) float __powf(float a, float b) ;

static __inline__ __attribute__((always_inline)) float fdividef(float a, float b);

static __inline__ __attribute__((always_inline)) double fdivide(double a, double b);

static __inline__ __attribute__((always_inline)) int __hadd(int a, int b);

static __inline__ __attribute__((always_inline)) int __rhadd(int a, int b);

static __inline__ __attribute__((always_inline)) unsigned int __uhadd(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __urhadd(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) float __fsub_rn (float a, float b);

static __inline__ __attribute__((always_inline)) float __fsub_rz (float a, float b);

static __inline__ __attribute__((always_inline)) float __fsub_rd (float a, float b);

static __inline__ __attribute__((always_inline)) float __fsub_ru (float a, float b);

static __inline__ __attribute__((always_inline)) float __frsqrt_rn (float a);

static __inline__ __attribute__((always_inline)) int __ffs(int a);

static __inline__ __attribute__((always_inline)) int __ffsll(long long int a);






static __inline__ __attribute__((always_inline))
int __iAtomicAdd(int *p, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicAdd_block(int *p, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicAdd_system(int *p, int val);
#line 3926 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAdd(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAdd_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAdd_system(unsigned int *p, unsigned int val);
#line 3937 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAdd(unsigned long long *p, unsigned long long val);


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAdd_block(unsigned long long *p, unsigned long long val);

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAdd_system(unsigned long long *p, unsigned long long val);
#line 3948 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
float __fAtomicAdd(float *p, float val);


static __inline__ __attribute__((always_inline))
float __fAtomicAdd_block(float *p, float val);

static __inline__ __attribute__((always_inline))
float __fAtomicAdd_system(float *p, float val);

static __inline__ __attribute__((always_inline))
double __dAtomicAdd(double *p, double val);

static __inline__ __attribute__((always_inline))
double __dAtomicAdd_block(double *p, double val);

static __inline__ __attribute__((always_inline))
double __dAtomicAdd_system(double *p, double val);
#line 3968 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
int __iAtomicExch(int *p, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicExch_block(int *p, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicExch_system(int *p, int val);
#line 3979 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicExch(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicExch_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicExch_system(unsigned int *p, unsigned int val);
#line 3990 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicExch(unsigned long long *p,
                                   unsigned long long val);


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicExch_block(unsigned long long *p, unsigned long long val);

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicExch_system(unsigned long long *p, unsigned long long val);
#line 4002 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
float __fAtomicExch(float *p, float val);


static __inline__ __attribute__((always_inline))
float __fAtomicExch_block(float *p, float val);

static __inline__ __attribute__((always_inline))
float __fAtomicExch_system(float *p, float val);
#line 4013 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
int __iAtomicMin(int *p, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicMin_block(int *p, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicMin_system(int *p, int val);
#line 4024 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long __illAtomicMin(long long *p, long long val);
#line 4029 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long __illAtomicMin_block(long long *p, long long val);

static __inline__ __attribute__((always_inline))
long long __illAtomicMin_system(long long *p, long long val);
#line 4037 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMin(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMin_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMin_system(unsigned int *p, unsigned int val);
#line 4048 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMin(unsigned long long *p, unsigned long long val);
#line 4053 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMin_block(unsigned long long *p, unsigned long long val);

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMin_system(unsigned long long *p, unsigned long long val);
#line 4061 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
int __iAtomicMax(int *p, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicMax_block(int *p, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicMax_system(int *p, int val);
#line 4072 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long __illAtomicMax(long long *p, long long val);
#line 4077 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long __illAtomicMax_block(long long *p, long long val);

static __inline__ __attribute__((always_inline))
long long __illAtomicMax_system(long long *p, long long val);
#line 4085 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMax(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMax_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMax_system(unsigned int *p, unsigned int val);
#line 4096 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMax(unsigned long long *p, unsigned long long val);
#line 4101 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMax_block(unsigned long long *p, unsigned long long val);

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMax_system(unsigned long long *p, unsigned long long val);
#line 4109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicInc(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicInc_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicInc_system(unsigned int *p, unsigned int val);
#line 4120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicDec(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicDec_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicDec_system(unsigned int *p, unsigned int val);
#line 4131 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
int __iAtomicCAS(int *p, int compare, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicCAS_block(int *p, int compare, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicCAS_system(int *p, int compare, int val);
#line 4142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicCAS(unsigned int *p, unsigned int compare,
                          unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicCAS_block(unsigned int *p, unsigned int compare,
                                unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicCAS_system(unsigned int *p, unsigned int compare,
                                 unsigned int val);
#line 4156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicCAS(unsigned long long int *p,
                                      unsigned long long int compare,
                                      unsigned long long int val);


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicCAS_block(unsigned long long int *p,
                                            unsigned long long int compare,
                                            unsigned long long int val);

static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicCAS_system(unsigned long long int *p,
                                             unsigned long long int compare,
                                             unsigned long long int val);
#line 4173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
int __iAtomicAnd(int *p, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicAnd_block(int *p, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicAnd_system(int *p, int val);
#line 4184 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long int __llAtomicAnd(long long int *p, long long int val);
#line 4189 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long __llAtomicAnd_block(long long *p, long long val);

static __inline__ __attribute__((always_inline))
long long __llAtomicAnd_system(long long *p, long long val);
#line 4197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAnd(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAnd_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAnd_system(unsigned int *p, unsigned int val);
#line 4208 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicAnd(unsigned long long int *p,
                                      unsigned long long int val);
#line 4214 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAnd_block(unsigned long long *p, unsigned long long val);

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAnd_system(unsigned long long *p, unsigned long long val);
#line 4222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
int __iAtomicOr(int *p, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicOr_block(int *p, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicOr_system(int *p, int val);
#line 4233 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long int __llAtomicOr(long long int *p, long long int val);
#line 4238 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long __llAtomicOr_block(long long *p, long long val);

static __inline__ __attribute__((always_inline))
long long __llAtomicOr_system(long long *p, long long val);
#line 4246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicOr(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicOr_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicOr_system(unsigned int *p, unsigned int val);
#line 4257 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicOr(unsigned long long int *p,
                                     unsigned long long int val);
#line 4263 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicOr_block(unsigned long long *p, unsigned long long val);

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicOr_system(unsigned long long *p, unsigned long long val);
#line 4271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
int __iAtomicXor(int *p, int val);


static __inline__ __attribute__((always_inline))
int __iAtomicXor_block(int *p, int val);

static __inline__ __attribute__((always_inline))
int __iAtomicXor_system(int *p, int val);
#line 4282 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long int __llAtomicXor(long long int *p, long long int val);
#line 4287 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
long long __llAtomicXor_block(long long *p, long long val);

static __inline__ __attribute__((always_inline))
long long __llAtomicXor_system(long long *p, long long val);
#line 4295 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicXor(unsigned int *p, unsigned int val);


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicXor_block(unsigned int *p, unsigned int val);

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicXor_system(unsigned int *p, unsigned int val);
#line 4306 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicXor(unsigned long long int *p,
                                      unsigned long long int val);
#line 4312 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicXor_block(unsigned long long *p, unsigned long long val);

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicXor_system(unsigned long long *p, unsigned long long val);
#line 4320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"







static __inline__ __attribute__((always_inline)) unsigned int __vabs2(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vabsss2(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vadd2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vaddss2 (unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vaddus2 (unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vavgs2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vavgu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vhaddu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpeq2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpges2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgeu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgts2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgtu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmples2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpleu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmplts2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpltu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpne2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vmaxs2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vmaxu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vmins2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vminu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vseteq2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetges2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetgeu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetgts2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetgtu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetles2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetleu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetlts2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetltu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetne2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsadu2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsub2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsubss2 (unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsubus2 (unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vneg2(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vnegss2(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffs2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsads2(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vabs4(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vabsss4(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vadd4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vaddss4 (unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vaddus4 (unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vavgs4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vavgu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vhaddu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpeq4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpges4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgeu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgts4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgtu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmples4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpleu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmplts4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpltu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vcmpne4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vmaxs4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vmaxu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vmins4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vminu4(unsigned int a, unsigned int b);
static __inline__ __attribute__((always_inline)) unsigned int __vseteq4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetles4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetleu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetlts4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetltu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetges4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetgeu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetgts4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetgtu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsetne4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsadu4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsub4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsubss4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsubus4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vneg4(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vnegss4(unsigned int a);

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffs4(unsigned int a, unsigned int b);

static __inline__ __attribute__((always_inline)) unsigned int __vsads4(unsigned int a, unsigned int b);








#line 4498 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"




#line 4503 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"








#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"








































































































































































































































#line 234 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"





#line 240 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


#line 243 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


#line 246 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"



#line 250 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"






static __inline__ __attribute__((always_inline)) int __syncthreads_count(int predicate)
{
  return __nvvm_bar0_popc(predicate);
}

static __inline__ __attribute__((always_inline)) int __syncthreads_and(int predicate)
{
  return __nvvm_bar0_and(predicate);
}

static __inline__ __attribute__((always_inline)) int __syncthreads_or(int predicate)
{
  return __nvvm_bar0_or(predicate);
}






static __inline__ __attribute__((always_inline)) void __threadfence_block()
{
  __nvvm_membar_cta();
}

static __inline__ __attribute__((always_inline)) void __threadfence()
{
  __nvvm_membar_gl();
}

static __inline__ __attribute__((always_inline)) void __threadfence_system()
{
  __nvvm_membar_sys();
}






static __inline__ __attribute__((always_inline)) int __all(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        ".reg .pred \t%%p2; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.all.pred \t%%p2, %%p1; \n\t"
        "selp.s32 \t%0, 1, 0, %%p2; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}

static __inline__ __attribute__((always_inline)) int __any(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        ".reg .pred \t%%p2; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.any.pred \t%%p2, %%p1; \n\t"
        "selp.s32 \t%0, 1, 0, %%p2; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}

static __inline__ __attribute__((always_inline))


#line 326 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
int
#line 328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
__ballot(int a)
{
  int result;
  asm __volatile__ ("{ \n\t"
        ".reg .pred \t%%p1; \n\t"
        "setp.ne.u32 \t%%p1, %1, 0; \n\t"
        "vote.ballot.b32 \t%0, %%p1; \n\t"
        "}" : "=r"(result) : "r"(a));
  return result;
}








#line 347 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline__ __attribute__((always_inline)) void __brkpt()
#line 349 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
{
  asm __volatile__ ("brkpt;");
}



#line 356 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline__ __attribute__((always_inline)) int clock()
#line 358 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
{
  int r;
  asm __volatile__ ("mov.u32 \t%0, %%clock;" : "=r"(r));
  return r;
}

static __inline__ __attribute__((always_inline)) long long clock64()
{
  long long z;
  asm __volatile__ ("mov.u64 \t%0, %%clock64;" : "=l"(z));
  return z;
}
    


static __inline__ __attribute__((always_inline)) unsigned int __pm0(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm0;" : "=r"(r));
  return r;
}

static __inline__ __attribute__((always_inline)) unsigned int __pm1(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm1;" : "=r"(r));
  return r;
}

static __inline__ __attribute__((always_inline)) unsigned int __pm2(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm2;" : "=r"(r));
  return r;
}

static __inline__ __attribute__((always_inline)) unsigned int __pm3(void)
{
  unsigned int r;
  asm("mov.u32 \t%0, %%pm3;" : "=r"(r));
  return r;
}

static __inline__ __attribute__((always_inline)) void __trap(void)
{
  asm __volatile__ ("trap;");
}

static __inline__ __attribute__((always_inline)) void* memcpy(void *dest, const void *src, size_t n) 
{


#line 411 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
  __nvvm_memcpy((unsigned char *)dest, (unsigned char *)src, n,
                 1);
  return dest;
#line 415 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
}

static __inline__ __attribute__((always_inline)) void* memset(void *dest, int c, size_t n) 
{


#line 422 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
  __nvvm_memset((unsigned char *)dest, (unsigned char)c, n,
                1);
  return dest;
#line 426 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
  
}






static __inline__ __attribute__((always_inline)) int __clz(int x)
{
  return __nv_clz(x);
}

static __inline__ __attribute__((always_inline)) int __clzll(long long x)
{
  return __nv_clzll(x);
}



#line 447 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline__ __attribute__((always_inline)) int __popc(int x)
#line 449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
{
  return __nv_popc(x);
}



#line 456 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline__ __attribute__((always_inline)) int __popcll(long long x)
#line 458 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
{
  return __nv_popcll(x);
}

static __inline__ __attribute__((always_inline)) unsigned int __byte_perm(unsigned int a,
                                                unsigned int b,
                                                unsigned int c)
{
  return __nv_byte_perm(a, b, c);
}






static __inline__ __attribute__((always_inline)) int min(int x, int y)
{
  return __nv_min(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned int umin(unsigned int x, unsigned int y)
{
  return __nv_umin(x, y);
}
    
static __inline__ __attribute__((always_inline)) long long llmin(long long x, long long y)
{
  return __nv_llmin(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned long long ullmin(unsigned long long x,
                                                 unsigned long long y)
{
  return __nv_ullmin(x, y);
}
    
static __inline__ __attribute__((always_inline)) int max(int x, int y)
{
  return __nv_max(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned int umax(unsigned int x, unsigned int y)
{
  return __nv_umax(x, y);
}
    
static __inline__ __attribute__((always_inline)) long long llmax(long long x, long long y)
{
  return __nv_llmax(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned long long ullmax(unsigned long long x,
                                                 unsigned long long y)
{
  return __nv_ullmax(x, y);
}

static __inline__ __attribute__((always_inline)) int __mulhi(int x, int y)
{
  return __nv_mulhi(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned int __umulhi(unsigned int x, unsigned int y)
{
  return __nv_umulhi(x, y);
}

static __inline__ __attribute__((always_inline)) long long __mul64hi(long long x, long long y)
{
  return __nv_mul64hi(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned long long __umul64hi(unsigned long long x,
                                                     unsigned long long y)
{
  return __nv_umul64hi(x, y);
}

static __inline__ __attribute__((always_inline)) int __mul24(int x, int y)
{
  return __nv_mul24(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned int __umul24(unsigned int x, unsigned int y)
{
  return __nv_umul24(x, y);
}

static __inline__ __attribute__((always_inline)) unsigned int __brev(unsigned int x)
{
  return __nv_brev(x);
}
    
static __inline__ __attribute__((always_inline)) unsigned long long __brevll(unsigned long long x)
{
  return __nv_brevll(x);
}
    


#line 560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static __inline__ __attribute__((always_inline)) int __sad(int x, int y, int z)
#line 562 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
{
  return __nv_sad(x, y, z);
}

static __inline__ __attribute__((always_inline)) unsigned int __usad(unsigned int x,
                                           unsigned int y,
                                           unsigned int z)
{
  return __nv_usad(x, y, z);
}

static __inline__ __attribute__((always_inline)) int abs(int x) 
{
  return __nv_abs(x);
}

static __inline__ __attribute__((always_inline)) long labs(long x) 
{


#line 583 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
  return __nv_abs((int) x);
#line 585 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
}

static __inline__ __attribute__((always_inline)) long long llabs(long long x) 
{
  return __nv_llabs(x);
}






static __inline__ __attribute__((always_inline)) float floorf(float f) 
{
  return __nv_floorf(f);
}

static __inline__ __attribute__((always_inline)) double floor(double f) 
{
  return __nv_floor(f);
}

static __inline__ __attribute__((always_inline)) float fabsf(float f) 
{
  return __nv_fabsf(f);
}

static __inline__ __attribute__((always_inline)) double fabs(double f) 
{
  return __nv_fabs(f);
}

static __inline__ __attribute__((always_inline)) float fminf(float x, float y) 
{
  return __nv_fminf(x, y);
}

static __inline__ __attribute__((always_inline)) float fmaxf(float x, float y) 
{
  return __nv_fmaxf(x, y);
}

static __inline__ __attribute__((always_inline)) float rsqrtf(float x)
{
  return __nv_rsqrtf(x);
}

static __inline__ __attribute__((always_inline)) double fmin(double x, double y) 
{
  return __nv_fmin(x, y);
}

static __inline__ __attribute__((always_inline)) double fmax(double x, double y) 
{
  return __nv_fmax(x, y);
}

static __inline__ __attribute__((always_inline)) double rsqrt(double x)
{
  return __nv_rsqrt(x);
}

static __inline__ __attribute__((always_inline)) double ceil(double x) 
{
  return __nv_ceil(x);
}

static __inline__ __attribute__((always_inline)) double trunc(double x) 
{
  return __nv_trunc(x);
}

static __inline__ __attribute__((always_inline)) float exp2f(float x)  
{
  return __nv_exp2f(x);
}

static __inline__ __attribute__((always_inline)) float truncf(float x) 
{
  return __nv_truncf(x);
}

static __inline__ __attribute__((always_inline)) float ceilf(float x) 
{
  return __nv_ceilf(x);
}

static __inline__ __attribute__((always_inline)) float __saturatef(float x)
{
  return __nv_saturatef(x);
}






static __inline__ __attribute__((always_inline)) float __fmaf_rn(float x, float y, float z)
{
  return __nv_fmaf_rn(x, y, z);
}

static __inline__ __attribute__((always_inline)) float __fmaf_rz(float x, float y, float z)
{
  return __nv_fmaf_rz(x, y, z);
}

static __inline__ __attribute__((always_inline)) float __fmaf_rd(float x, float y, float z)
{
  return __nv_fmaf_rd(x, y, z);
}

static __inline__ __attribute__((always_inline)) float __fmaf_ru(float x, float y, float z)
{
  return __nv_fmaf_ru(x, y, z);
}






static __inline__ __attribute__((always_inline)) float __fmaf_ieee_rn(float x, float y, float z)
{
  return __nv_fmaf_ieee_rn(x, y, z);
}

static __inline__ __attribute__((always_inline)) float __fmaf_ieee_rz(float x, float y, float z)
{
  return __nv_fmaf_ieee_rz(x, y, z);
}

static __inline__ __attribute__((always_inline)) float __fmaf_ieee_rd(float x, float y, float z)
{
  return __nv_fmaf_ieee_rd(x, y, z);
}

static __inline__ __attribute__((always_inline)) float __fmaf_ieee_ru(float x, float y, float z)
{
  return __nv_fmaf_ieee_ru(x, y, z);
}






static __inline__ __attribute__((always_inline)) double __fma_rn(double x, double y, double z)
{
  return __nv_fma_rn(x, y, z);
}

static __inline__ __attribute__((always_inline)) double __fma_rz(double x, double y, double z)
{
  return __nv_fma_rz(x, y, z);
}

static __inline__ __attribute__((always_inline)) double __fma_rd(double x, double y, double z)
{
  return __nv_fma_rd(x, y, z);
}

static __inline__ __attribute__((always_inline)) double __fma_ru(double x, double y, double z)
{
  return __nv_fma_ru(x, y, z);
}

static __inline__ __attribute__((always_inline)) float __fdividef(float x, float y)
{
  return __nv_fast_fdividef(x, y);
}






static __inline__ __attribute__((always_inline)) float __fdiv_rn(float x, float y)
{
  return __nv_fdiv_rn(x, y);
}

static __inline__ __attribute__((always_inline)) float __fdiv_rz(float x, float y)
{
  return __nv_fdiv_rz(x, y);
}

static __inline__ __attribute__((always_inline)) float __fdiv_rd(float x, float y)
{
  return __nv_fdiv_rd(x, y);
}

static __inline__ __attribute__((always_inline)) float __fdiv_ru(float x, float y)
{
  return __nv_fdiv_ru(x, y);
}






static __inline__ __attribute__((always_inline)) float __frcp_rn(float x)
{
  return __nv_frcp_rn(x);
}

static __inline__ __attribute__((always_inline)) float __frcp_rz(float x)
{
  return __nv_frcp_rz(x);
}

static __inline__ __attribute__((always_inline)) float __frcp_rd(float x)
{
  return __nv_frcp_rd(x);
}

static __inline__ __attribute__((always_inline)) float __frcp_ru(float x)
{
  return __nv_frcp_ru(x);
}






static __inline__ __attribute__((always_inline)) float __fsqrt_rn(float x)
{
  return __nv_fsqrt_rn(x);
}

static __inline__ __attribute__((always_inline)) float __fsqrt_rz(float x)
{
  return __nv_fsqrt_rz(x);
}

static __inline__ __attribute__((always_inline)) float __fsqrt_rd(float x)
{
  return __nv_fsqrt_rd(x);
}

static __inline__ __attribute__((always_inline)) float __fsqrt_ru(float x)
{
  return __nv_fsqrt_ru(x);
}






static __inline__ __attribute__((always_inline)) double __ddiv_rn(double x, double y)
{
  return __nv_ddiv_rn(x, y);
}

static __inline__ __attribute__((always_inline)) double __ddiv_rz(double x, double y)
{
  return __nv_ddiv_rz(x, y);
}

static __inline__ __attribute__((always_inline)) double __ddiv_rd(double x, double y)
{
  return __nv_ddiv_rd(x, y);
}

static __inline__ __attribute__((always_inline)) double __ddiv_ru(double x, double y)
{
  return __nv_ddiv_ru(x, y);
}






static __inline__ __attribute__((always_inline)) double __drcp_rn(double x)
{
  return __nv_drcp_rn(x);
}

static __inline__ __attribute__((always_inline)) double __drcp_rz(double x)
{
  return __nv_drcp_rz(x);
}

static __inline__ __attribute__((always_inline)) double __drcp_rd(double x)
{
  return __nv_drcp_rd(x);
}

static __inline__ __attribute__((always_inline)) double __drcp_ru(double x)
{
  return __nv_drcp_ru(x);
}






static __inline__ __attribute__((always_inline)) double __dsqrt_rn(double x)
{
  return __nv_dsqrt_rn(x);
}

static __inline__ __attribute__((always_inline)) double __dsqrt_rz(double x)
{
  return __nv_dsqrt_rz(x);
}

static __inline__ __attribute__((always_inline)) double __dsqrt_rd(double x)
{
  return __nv_dsqrt_rd(x);
}

static __inline__ __attribute__((always_inline)) double __dsqrt_ru(double x)
{
  return __nv_dsqrt_ru(x);
}

static __inline__ __attribute__((always_inline)) float sqrtf(float x) 
{
  return __nv_sqrtf(x);
}

static __inline__ __attribute__((always_inline)) double sqrt(double x) 
{
  return __nv_sqrt(x);
}






static __inline__ __attribute__((always_inline)) double __dadd_rn(double x, double y)
{
  return __nv_dadd_rn(x, y);
}

static __inline__ __attribute__((always_inline)) double __dadd_rz(double x, double y)
{
  return __nv_dadd_rz(x, y);
}

static __inline__ __attribute__((always_inline)) double __dadd_rd(double x, double y)
{
  return __nv_dadd_rd(x, y);
}

static __inline__ __attribute__((always_inline)) double __dadd_ru(double x, double y)
{
  return __nv_dadd_ru(x, y);
}






static __inline__ __attribute__((always_inline)) double __dmul_rn(double x, double y)
{
  return __nv_dmul_rn(x, y);
}

static __inline__ __attribute__((always_inline)) double __dmul_rz(double x, double y)
{
  return __nv_dmul_rz(x, y);
}

static __inline__ __attribute__((always_inline)) double __dmul_rd(double x, double y)
{
  return __nv_dmul_rd(x, y);
}

static __inline__ __attribute__((always_inline)) double __dmul_ru(double x, double y)
{
  return __nv_dmul_ru(x, y);
}






static __inline__ __attribute__((always_inline)) float __fadd_rd(float x, float y)
{
  return __nv_fadd_rd(x, y);
}

static __inline__ __attribute__((always_inline)) float __fadd_ru(float x, float y)
{
  return __nv_fadd_ru(x, y);
}

static __inline__ __attribute__((always_inline)) float __fadd_rn(float x, float y)
{
  return __nv_fadd_rn(x, y);
}

static __inline__ __attribute__((always_inline)) float __fadd_rz(float x, float y)
{
  return __nv_fadd_rz(x, y);
}






static __inline__ __attribute__((always_inline)) float __fmul_rd(float x, float y)
{
  return __nv_fmul_rd(x, y);
}

static __inline__ __attribute__((always_inline)) float __fmul_ru(float x, float y)
{
  return __nv_fmul_ru(x, y);
}

static __inline__ __attribute__((always_inline)) float __fmul_rn(float x, float y)
{
  return __nv_fmul_rn(x, y);
}

static __inline__ __attribute__((always_inline)) float __fmul_rz(float x, float y)
{
  return __nv_fmul_rz(x, y);
}







static __inline__ __attribute__((always_inline)) float __double2float_rn(double d)
{
  return __nv_double2float_rn(d);
}

static __inline__ __attribute__((always_inline)) float __double2float_rz(double d)
{
  return __nv_double2float_rz(d);
}

static __inline__ __attribute__((always_inline)) float __double2float_rd(double d)
{
  return __nv_double2float_rd(d);
}

static __inline__ __attribute__((always_inline)) float __double2float_ru(double d)
{
  return __nv_double2float_ru(d);
}
    

static __inline__ __attribute__((always_inline)) int __double2int_rn(double d)
{
  return __nv_double2int_rn(d);
}

static __inline__ __attribute__((always_inline)) int __double2int_rz(double d)
{
  return __nv_double2int_rz(d);
}

static __inline__ __attribute__((always_inline)) int __double2int_rd(double d)
{
  return __nv_double2int_rd(d);
}

static __inline__ __attribute__((always_inline)) int __double2int_ru(double d)
{
  return __nv_double2int_ru(d);
}


static __inline__ __attribute__((always_inline)) unsigned int __double2uint_rn(double d)
{
  return __nv_double2uint_rn(d);
}

static __inline__ __attribute__((always_inline)) unsigned int __double2uint_rz(double d)
{
  return __nv_double2uint_rz(d);
}

static __inline__ __attribute__((always_inline)) unsigned int __double2uint_rd(double d)
{
  return __nv_double2uint_rd(d);
}

static __inline__ __attribute__((always_inline)) unsigned int __double2uint_ru(double d)
{
  return __nv_double2uint_ru(d);
}


static __inline__ __attribute__((always_inline)) double __int2double_rn(int i)
{
  return __nv_int2double_rn(i);
}


static __inline__ __attribute__((always_inline)) double __uint2double_rn(unsigned int i)
{
  return __nv_uint2double_rn(i);
}


static __inline__ __attribute__((always_inline)) int __float2int_rn(float in)
{
  return __nv_float2int_rn(in);
}

static __inline__ __attribute__((always_inline)) int __float2int_rz(float in)
{
  return __nv_float2int_rz(in);
}

static __inline__ __attribute__((always_inline)) int __float2int_rd(float in)
{
  return __nv_float2int_rd(in);
}

static __inline__ __attribute__((always_inline)) int __float2int_ru(float in)
{
  return __nv_float2int_ru(in);
}


static __inline__ __attribute__((always_inline)) unsigned int __float2uint_rn(float in)
{
  return __nv_float2uint_rn(in);
}

static __inline__ __attribute__((always_inline)) unsigned int __float2uint_rz(float in)
{
  return __nv_float2uint_rz(in);
}

static __inline__ __attribute__((always_inline)) unsigned int __float2uint_rd(float in)
{
  return __nv_float2uint_rd(in);
}

static __inline__ __attribute__((always_inline)) unsigned int __float2uint_ru(float in)
{
  return __nv_float2uint_ru(in);
}


static __inline__ __attribute__((always_inline)) float __int2float_rn(int in)
{
  return __nv_int2float_rn(in);
}

static __inline__ __attribute__((always_inline)) float __int2float_rz(int in)
{
  return __nv_int2float_rz(in);
}

static __inline__ __attribute__((always_inline)) float __int2float_rd(int in)
{
  return __nv_int2float_rd(in);
}

static __inline__ __attribute__((always_inline)) float __int2float_ru(int in)
{
  return __nv_int2float_ru(in);
}


static __inline__ __attribute__((always_inline)) float __uint2float_rn(unsigned int in)
{
  return __nv_uint2float_rn(in);
}

static __inline__ __attribute__((always_inline)) float __uint2float_rz(unsigned int in)
{
  return __nv_uint2float_rz(in);
}

static __inline__ __attribute__((always_inline)) float __uint2float_rd(unsigned int in)
{
  return __nv_uint2float_rd(in);
}

static __inline__ __attribute__((always_inline)) float __uint2float_ru(unsigned int in)
{
  return __nv_uint2float_ru(in);
}


static __inline__ __attribute__((always_inline)) double __hiloint2double(int a, int b)
{
  return __nv_hiloint2double(a, b);
}

static __inline__ __attribute__((always_inline)) int __double2loint(double d)
{
  return __nv_double2loint(d);
}

static __inline__ __attribute__((always_inline)) int __double2hiint(double d)
{
  return __nv_double2hiint(d);
}


static __inline__ __attribute__((always_inline)) long long __float2ll_rn(float f)
{
  return __nv_float2ll_rn(f);
}

static __inline__ __attribute__((always_inline)) long long __float2ll_rz(float f)
{
  return __nv_float2ll_rz(f);
}

static __inline__ __attribute__((always_inline)) long long __float2ll_rd(float f)
{
  return __nv_float2ll_rd(f);
}

static __inline__ __attribute__((always_inline)) long long __float2ll_ru(float f)
{
  return __nv_float2ll_ru(f);
}


static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_rn(float f)
{
  return __nv_float2ull_rn(f);
}

static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_rz(float f)
{
  return __nv_float2ull_rz(f);
}

static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_rd(float f)
{
  return __nv_float2ull_rd(f);
}

static __inline__ __attribute__((always_inline)) unsigned long long __float2ull_ru(float f)
{
  return __nv_float2ull_ru(f);
}


static __inline__ __attribute__((always_inline)) long long __double2ll_rn(double f)
{
  return __nv_double2ll_rn(f);
}

static __inline__ __attribute__((always_inline)) long long __double2ll_rz(double f)
{
  return __nv_double2ll_rz(f);
}

static __inline__ __attribute__((always_inline)) long long __double2ll_rd(double f)
{
  return __nv_double2ll_rd(f);
}

static __inline__ __attribute__((always_inline)) long long __double2ll_ru(double f)
{
  return __nv_double2ll_ru(f);
}


static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_rn(double f)
{
  return __nv_double2ull_rn(f);
}

static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_rz(double f)
{
  return __nv_double2ull_rz(f);
}

static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_rd(double f)
{
  return __nv_double2ull_rd(f);
}

static __inline__ __attribute__((always_inline)) unsigned long long __double2ull_ru(double f)
{
  return __nv_double2ull_ru(f);
}


static __inline__ __attribute__((always_inline)) float __ll2float_rn(long long l)
{
  return __nv_ll2float_rn(l);
}

static __inline__ __attribute__((always_inline)) float __ll2float_rz(long long l)
{
  return __nv_ll2float_rz(l);
}

static __inline__ __attribute__((always_inline)) float __ll2float_rd(long long l)
{
  return __nv_ll2float_rd(l);
}

static __inline__ __attribute__((always_inline)) float __ll2float_ru(long long l)
{
  return __nv_ll2float_ru(l);
}


static __inline__ __attribute__((always_inline)) float __ull2float_rn(unsigned long long l)
{
  return __nv_ull2float_rn(l);
}

static __inline__ __attribute__((always_inline)) float __ull2float_rz(unsigned long long l)
{
  return __nv_ull2float_rz(l);
}

static __inline__ __attribute__((always_inline)) float __ull2float_rd(unsigned long long l)
{
  return __nv_ull2float_rd(l);
}

static __inline__ __attribute__((always_inline)) float __ull2float_ru(unsigned long long l)
{
  return __nv_ull2float_ru(l);
}


static __inline__ __attribute__((always_inline)) double __ll2double_rn(long long l)
{
  return __nv_ll2double_rn(l);
}

static __inline__ __attribute__((always_inline)) double __ll2double_rz(long long l)
{
  return __nv_ll2double_rz(l);
}

static __inline__ __attribute__((always_inline)) double __ll2double_rd(long long l)
{
  return __nv_ll2double_rd(l);
}

static __inline__ __attribute__((always_inline)) double __ll2double_ru(long long l)
{
  return __nv_ll2double_ru(l);
}


static __inline__ __attribute__((always_inline)) double __ull2double_rn(unsigned long long l)
{
  return __nv_ull2double_rn(l);
}

static __inline__ __attribute__((always_inline)) double __ull2double_rz(unsigned long long l)
{
  return __nv_ull2double_rz(l);
}

static __inline__ __attribute__((always_inline)) double __ull2double_rd(unsigned long long l)
{
  return __nv_ull2double_rd(l);
}

static __inline__ __attribute__((always_inline)) double __ull2double_ru(unsigned long long l)
{
  return __nv_ull2double_ru(l);
}

static __inline__ __attribute__((always_inline)) unsigned short __float2half_rn(float f)
{
  return __nv_float2half_rn(f);
}

static __inline__ __attribute__((always_inline)) float __half2float(unsigned short h)
{
  return __nv_half2float(h);
}

static __inline__ __attribute__((always_inline)) float __int_as_float(int x)
{
  return __nv_int_as_float(x);
}

static __inline__ __attribute__((always_inline)) int __float_as_int(float x)
{
  return __nv_float_as_int(x);
}

static __inline__ __attribute__((always_inline)) float __uint_as_float(unsigned int x)
{
  return __nv_uint_as_float(x);
}

static __inline__ __attribute__((always_inline)) unsigned int __float_as_uint(float x)
{
  return __nv_float_as_uint(x);
}
    
static __inline__ __attribute__((always_inline)) double __longlong_as_double(long long x)
{
  return __nv_longlong_as_double(x);
}

static __inline__ __attribute__((always_inline)) long long  __double_as_longlong (double x)
{
  return __nv_double_as_longlong(x);
}







static __inline__ __attribute__((always_inline)) float __sinf(float a) 
{
  return __nv_fast_sinf(a);
}

static __inline__ __attribute__((always_inline)) float __cosf(float a) 
{
  return __nv_fast_cosf(a);
}

static __inline__ __attribute__((always_inline)) float __log2f(float a) 
{
  return __nv_fast_log2f(a);
}







static __inline__ __attribute__((always_inline)) float __tanf(float a) 
{
  return __nv_fast_tanf(a);
}

static __inline__ __attribute__((always_inline)) void __sincosf(float a, float *sptr, float *cptr) 
{
  __nv_fast_sincosf(a, sptr, cptr);
}

static __inline__ __attribute__((always_inline)) float __expf(float a) 
{
  return __nv_fast_expf(a);
}

static __inline__ __attribute__((always_inline)) float __exp10f(float a) 
{
  return __nv_fast_exp10f(a);
}

static __inline__ __attribute__((always_inline)) float __log10f(float a) 
{
  return __nv_fast_log10f(a);
}

static __inline__ __attribute__((always_inline)) float __logf(float a) 
{
  return __nv_fast_logf(a);
}

static __inline__ __attribute__((always_inline)) float __powf(float a, float b) 
{
  return __nv_fast_powf(a, b);
}

static __inline__ __attribute__((always_inline)) float fdividef(float a, float b)
{
  if (1 && !0) {
    return __nv_fast_fdividef(a, b);
  } else {
    return a / b;
  }
}

static __inline__ __attribute__((always_inline)) double fdivide(double a, double b)
{
  return a / b;
}

static __inline__ __attribute__((always_inline)) int __hadd(int a, int b)
{
  return __nv_hadd(a, b);
}

static __inline__ __attribute__((always_inline)) int __rhadd(int a, int b)
{
  return __nv_rhadd(a, b);
}

static __inline__ __attribute__((always_inline)) unsigned int __uhadd(unsigned int a, unsigned int b)
{
  return __nv_uhadd(a, b);
}

static __inline__ __attribute__((always_inline)) unsigned int __urhadd(unsigned int a, unsigned int b)
{
  return __nv_urhadd(a, b);
}

static __inline__ __attribute__((always_inline)) float __fsub_rn (float a, float b)
{
  return __nv_fsub_rn(a, b);
}

static __inline__ __attribute__((always_inline)) float __fsub_rz (float a, float b)
{
  return __nv_fsub_rz(a, b);
}

static __inline__ __attribute__((always_inline)) float __fsub_rd (float a, float b)
{
  return __nv_fsub_rd(a, b);
}

static __inline__ __attribute__((always_inline)) float __fsub_ru (float a, float b)
{
  return __nv_fsub_ru(a, b);
}

static __inline__ __attribute__((always_inline)) float __frsqrt_rn (float a)
{
  return __nv_frsqrt_rn(a);
}

static __inline__ __attribute__((always_inline)) int __ffs(int a)
{
  return __nv_ffs(a);
}

static __inline__ __attribute__((always_inline)) int __ffsll(long long int a)
{
  return __nv_ffsll(a);
}






static __inline__ __attribute__((always_inline))
int __iAtomicAdd(int *p, int val)
{
  return __nvvm_atom_add_gen_i((volatile int *)p, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicAdd_block(int *p, int val)
{
  return __nvvm_atom_cta_add_gen_i((volatile int *)p, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicAdd_system(int *p, int val)
{
  return __nvvm_atom_sys_add_gen_i((volatile int *)p, val);
}
#line 1560 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAdd(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_add_gen_i((volatile int *)p, (int)val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAdd_block(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_cta_add_gen_i((volatile int *)p, (int)val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAdd_system(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_sys_add_gen_i((volatile int *)p, (int)val);
}
#line 1580 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAdd(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_add_gen_ll((volatile long long *)p, (long long)val);
}


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAdd_block(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_cta_add_gen_ll((volatile long long *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAdd_system(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_sys_add_gen_ll((volatile long long *)p, (long long)val);
}
#line 1600 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
float __fAtomicAdd(float *p, float val)
{
  return __nvvm_atom_add_gen_f((volatile float *)p, val);
}


static __inline__ __attribute__((always_inline))
float __fAtomicAdd_block(float *p, float val)
{
  return __nvvm_atom_cta_add_gen_f((volatile float *)p, val);
}

static __inline__ __attribute__((always_inline))
float __fAtomicAdd_system(float *p, float val)
{
  return __nvvm_atom_sys_add_gen_f((volatile float *)p, val);
}

static __inline__ __attribute__((always_inline))
double __dAtomicAdd(double *p, double val)
{
  return __nvvm_atom_add_gen_d((volatile double *)p, val);
}

static __inline__ __attribute__((always_inline))
double __dAtomicAdd_block(double *p, double val)
{
  return __nvvm_atom_cta_add_gen_d((volatile double *)p, val);
}

static __inline__ __attribute__((always_inline))
double __dAtomicAdd_system(double *p, double val)
{
  return __nvvm_atom_sys_add_gen_d((volatile double *)p, val);
}
#line 1638 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
int __iAtomicExch(int *p, int val)
{
  return __nvvm_atom_xchg_gen_i((volatile int *)p, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicExch_block(int *p, int val)
{
  return __nvvm_atom_cta_xchg_gen_i((volatile int *)p, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicExch_system(int *p, int val)
{
  return __nvvm_atom_sys_xchg_gen_i((volatile int *)p, val);
}
#line 1659 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicExch(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_xchg_gen_i((volatile int *)p, (int)val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicExch_block(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_cta_xchg_gen_i((volatile int *)p, (int)val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicExch_system(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_sys_xchg_gen_i((volatile int *)p, (int)val);
}
#line 1679 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicExch(unsigned long long *p,
                                   unsigned long long val)
{
  return __nvvm_atom_xchg_gen_ll((volatile long long *)p, (long long)val);
}


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicExch_block(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_cta_xchg_gen_ll((volatile long long *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicExch_system(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_sys_xchg_gen_ll((volatile long long *)p, (long long)val);
}
#line 1700 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
float __fAtomicExch(float *p, float val)
{
  int old = __nvvm_atom_xchg_gen_i((volatile int *)p, __float_as_int(val));
  return __int_as_float(old);
}


static __inline__ __attribute__((always_inline))
float __fAtomicExch_block(float *p, float val)
{
  int old = __nvvm_atom_cta_xchg_gen_i((volatile int *)p, __float_as_int(val));
  return __int_as_float(old);
}

static __inline__ __attribute__((always_inline))
float __fAtomicExch_system(float *p, float val)
{
  int old = __nvvm_atom_sys_xchg_gen_i((volatile int *)p, __float_as_int(val));
  return __int_as_float(old);
}
#line 1723 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
int __iAtomicMin(int *p, int val)
{
  return __nvvm_atom_min_gen_i((volatile int *)p, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicMin_block(int *p, int val)
{
  return __nvvm_atom_cta_min_gen_i((volatile int *)p, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicMin_system(int *p, int val)
{
  return __nvvm_atom_sys_min_gen_i((volatile int *)p, val);
}
#line 1743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long __illAtomicMin(long long *p, long long val)
{
  return __nvvm_atom_min_gen_ll((volatile long long *)p, val);
}
#line 1751 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long __illAtomicMin_block(long long *p, long long val)
{
  return __nvvm_atom_cta_min_gen_ll((volatile long long *)p, val);
}

static __inline__ __attribute__((always_inline))
long long __illAtomicMin_system(long long *p, long long val)
{
  return __nvvm_atom_sys_min_gen_ll((volatile long long *)p, val);
}
#line 1765 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMin(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_min_gen_ui((volatile unsigned int *)p, val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMin_block(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_cta_min_gen_ui((volatile unsigned int *)p, val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMin_system(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_sys_min_gen_ui((volatile unsigned int *)p, val);
}
#line 1785 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMin(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_min_gen_ull((volatile unsigned long long *)p, val);
}
#line 1793 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMin_block(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_cta_min_gen_ull((volatile unsigned long long *)p, val);
}

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMin_system(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_sys_min_gen_ull((volatile unsigned long long *)p, val);
}
#line 1807 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
int __iAtomicMax(int *p, int val)
{
  return __nvvm_atom_max_gen_i((volatile int *)p, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicMax_block(int *p, int val)
{
  return __nvvm_atom_cta_max_gen_i((volatile int *)p, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicMax_system(int *p, int val)
{
  return __nvvm_atom_sys_max_gen_i((volatile int *)p, val);
}
#line 1827 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long __illAtomicMax(long long *p, long long val)
{
  return __nvvm_atom_max_gen_ll((volatile long long *)p, val);
}
#line 1835 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long __illAtomicMax_block(long long *p, long long val)
{
  return __nvvm_atom_cta_max_gen_ll((volatile long long *)p, val);
}

static __inline__ __attribute__((always_inline))
long long __illAtomicMax_system(long long *p, long long val)
{
  return __nvvm_atom_sys_max_gen_ll((volatile long long *)p, val);
}
#line 1849 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMax(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_max_gen_ui((unsigned int *)p, val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMax_block(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_cta_max_gen_ui((volatile unsigned int *)p, val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicMax_system(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_sys_max_gen_ui((volatile unsigned int *)p, val);
}
#line 1869 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMax(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_max_gen_ull((volatile unsigned long long *)p, val);
}
#line 1877 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMax_block(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_cta_max_gen_ull((volatile unsigned long long *)p, val);
}

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicMax_system(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_sys_max_gen_ull((volatile unsigned long long *)p, val);
}
#line 1891 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicInc(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_inc_gen_ui((unsigned int *)p, val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicInc_block(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_cta_inc_gen_ui((volatile unsigned int *)p, val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicInc_system(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_sys_inc_gen_ui((volatile unsigned int *)p, val);
}
#line 1911 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicDec(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_dec_gen_ui((unsigned int *)p, val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicDec_block(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_cta_dec_gen_ui((volatile unsigned int *)p, val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicDec_system(unsigned int *p, unsigned int val)
{
  return __nvvm_atom_sys_dec_gen_ui((volatile unsigned int *)p, val);
}
#line 1931 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
int __iAtomicCAS(int *p, int compare, int val)
{
  return __nvvm_atom_cas_gen_i((int *)p, compare, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicCAS_block(int *p, int compare, int val)
{
  return __nvvm_atom_cta_cas_gen_i((int *)p, compare, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicCAS_system(int *p, int compare, int val)
{
  return __nvvm_atom_sys_cas_gen_i((int *)p, compare, val);
}
#line 1951 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicCAS(unsigned int *p, unsigned int compare,
                          unsigned int val)
{
  return (unsigned int)__nvvm_atom_cas_gen_i((volatile int *)p,
                                             (int)compare,
                                             (int)val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicCAS_block(unsigned int *p, unsigned int compare,
                                unsigned int val)
{
  return (unsigned int)__nvvm_atom_cta_cas_gen_i((volatile int *)p,
                                                 (int)compare,
                                                 (int)val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicCAS_system(unsigned int *p, unsigned int compare,
                                 unsigned int val)
{
  return (unsigned int)__nvvm_atom_sys_cas_gen_i((volatile int *)p,
                                                 (int)compare,
                                                 (int)val);
}
#line 1980 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicCAS(unsigned long long int *p,
                                      unsigned long long int compare,
                                      unsigned long long int val)
{
  return
    (unsigned long long int)__nvvm_atom_cas_gen_ll((volatile long long int *)p,
                                                   (long long int)compare,
                                                   (long long int)val);
}


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicCAS_block(unsigned long long int *p,
                                            unsigned long long int compare,
                                            unsigned long long int val)
{
  return
    (unsigned long long int)__nvvm_atom_cta_cas_gen_ll((volatile long long int *)p,
                                                       (long long int)compare,
                                                       (long long int)val);
}

static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicCAS_system(unsigned long long int *p,
                                             unsigned long long int compare,
                                             unsigned long long int val)
{
  return
    (unsigned long long int)__nvvm_atom_sys_cas_gen_ll((volatile long long int *)p,
                                                       (long long int)compare,
                                                       (long long int)val);
}
#line 2015 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
int __iAtomicAnd(int *p, int val)
{
  return __nvvm_atom_and_gen_i((volatile int *)p, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicAnd_block(int *p, int val)
{
  return __nvvm_atom_cta_and_gen_i((volatile int *)p, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicAnd_system(int *p, int val)
{
  return __nvvm_atom_sys_and_gen_i((volatile int *)p, val);
}
#line 2035 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long int __llAtomicAnd(long long int *p, long long int val)
{
  return __nvvm_atom_and_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2043 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long int __llAtomicAnd_block(long long int *p, long long int val)
{
  return __nvvm_atom_cta_and_gen_ll((volatile long long int *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
long long int __llAtomicAnd_system(long long int *p, long long int val)
{
  return __nvvm_atom_sys_and_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2057 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAnd(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_and_gen_i((volatile int *)p, (int)val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAnd_block(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_cta_and_gen_i((volatile int *)p, (int)val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicAnd_system(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_sys_and_gen_i((volatile int *)p, (int)val);
}
#line 2077 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicAnd(unsigned long long int *p,
                                      unsigned long long int val)
{
  return __nvvm_atom_and_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2086 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAnd_block(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_cta_and_gen_ll((volatile long long *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicAnd_system(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_sys_and_gen_ll((volatile long long *)p, (long long)val);
}
#line 2100 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
int __iAtomicOr(int *p, int val)
{
  return __nvvm_atom_or_gen_i((volatile int *)p, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicOr_block(int *p, int val)
{
  return __nvvm_atom_cta_or_gen_i((volatile int *)p, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicOr_system(int *p, int val)
{
  return __nvvm_atom_sys_or_gen_i((volatile int *)p, val);
}
#line 2120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long int __llAtomicOr(long long int *p, long long int val)
{
  return __nvvm_atom_or_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long int __llAtomicOr_block(long long int *p, long long int val)
{
  return __nvvm_atom_cta_or_gen_ll((volatile long long int *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
long long int __llAtomicOr_system(long long int *p, long long int val)
{
  return __nvvm_atom_sys_or_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2142 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicOr(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_or_gen_i((volatile int *)p, (int)val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicOr_block(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_cta_or_gen_i((volatile int *)p, (int)val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicOr_system(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_sys_or_gen_i((volatile int *)p, (int)val);
}
#line 2162 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicOr(unsigned long long int *p,
                                     unsigned long long int val)
{
  return __nvvm_atom_or_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicOr_block(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_cta_or_gen_ll((volatile long long *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicOr_system(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_sys_or_gen_ll((volatile long long *)p, (long long)val);
}
#line 2185 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
int __iAtomicXor(int *p, int val)
{
  return __nvvm_atom_xor_gen_i((volatile int *)p, val);
}


static __inline__ __attribute__((always_inline))
int __iAtomicXor_block(int *p, int val)
{
  return __nvvm_atom_cta_xor_gen_i((volatile int *)p, val);
}

static __inline__ __attribute__((always_inline))
int __iAtomicXor_system(int *p, int val)
{
  return __nvvm_atom_sys_xor_gen_i((volatile int *)p, val);
}
#line 2205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long int __llAtomicXor(long long int *p, long long int val)
{
  return __nvvm_atom_xor_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2213 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
long long int __llAtomicXor_block(long long int *p, long long int val)
{
  return __nvvm_atom_cta_xor_gen_ll((volatile long long int *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
long long int __llAtomicXor_system(long long int *p, long long int val)
{
  return __nvvm_atom_sys_xor_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2227 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicXor(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_xor_gen_i((volatile int *)p, (int)val);
}


static __inline__ __attribute__((always_inline))
unsigned int __uAtomicXor_block(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_cta_xor_gen_i((volatile int *)p, (int)val);
}

static __inline__ __attribute__((always_inline))
unsigned int __uAtomicXor_system(unsigned int *p, unsigned int val)
{
  return (unsigned int)__nvvm_atom_sys_xor_gen_i((volatile int *)p, (int)val);
}
#line 2247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long int __ullAtomicXor(unsigned long long int *p,
                                      unsigned long long int val)
{
  return __nvvm_atom_xor_gen_ll((volatile long long int *)p, (long long)val);
}
#line 2256 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicXor_block(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_cta_xor_gen_ll((volatile long long *)p, (long long)val);
}

static __inline__ __attribute__((always_inline))
unsigned long long __ullAtomicXor_system(unsigned long long *p, unsigned long long val)
{
  return __nvvm_atom_sys_xor_gen_ll((volatile long long *)p, (long long)val);
}
#line 2270 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"







static __inline__ __attribute__((always_inline)) unsigned int __vabs2(unsigned int a)
{
    unsigned int r;

    unsigned int b = 0, c = 0;
    asm ("vabsdiff2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));











#line 2295 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vabsss2(unsigned int a)
{
    unsigned int r;

    unsigned int b = 0, c = 0;
    asm("vabsdiff2.s32.s32.s32.sat %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(c));














#line 2319 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vadd2(unsigned int a, unsigned int b)
{
    unsigned int s, t;

    s = 0;
    asm ("vadd2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(t) : "r"(a), "r"(b), "r"(s));






#line 2335 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return t;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vaddss2 (unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vadd2.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));














#line 2359 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vaddus2 (unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vadd2.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));


















#line 2387 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vavgs2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vavrg2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));


































#line 2431 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vavgu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vavrg2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));








#line 2449 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vhaddu2(unsigned int a, unsigned int b)
{
    
    
    unsigned int r, s;
    s = a ^ b;
    r = a & b;
    s = s & 0xfffefffe; 
    s = s >> 1;
    r = r + s;
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpeq2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          









#line 2483 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpges2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          















#line 2510 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgeu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          




#line 2526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgts2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          















#line 2553 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgtu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          




#line 2569 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmples2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          















#line 2596 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpleu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          




#line 2612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmplts2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          















#line 2639 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpltu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          




#line 2655 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpne2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 16;        
    r = c - r;          








#line 2675 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vabsdiff2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));













#line 2698 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vmaxs2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmax2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));









#line 2717 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vmaxu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmax2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));









#line 2736 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vmins2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmin2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));









#line 2755 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vminu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmin2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));









#line 2774 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vseteq2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));









#line 2793 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetges2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
















#line 2819 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetgeu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 2834 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetgts2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
















#line 2860 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetgtu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 2875 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetles2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
















#line 2901 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetleu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 2916 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetlts2(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset2.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
















#line 2942 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetltu2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 2957 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetne2(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset2.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));









#line 2976 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsadu2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm("vabsdiff2.u32.u32.u32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));


















#line 3004 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsub2(unsigned int a, unsigned int b)
{
    unsigned int s, t;

    s = 0;
    asm ("vsub2.u32.u32.u32 %0,%1,%2,%3;" : "=r"(t) : "r"(a), "r"(b), "r"(s));






#line 3020 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return t;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsubss2 (unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vsub2.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));














#line 3044 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsubus2 (unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vsub2.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));


















#line 3072 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vneg2(unsigned int a)
{
    return __vsub2 (0, a);
}

static __inline__ __attribute__((always_inline)) unsigned int __vnegss2(unsigned int a)
{
    return __vsubss2(0,a);
}

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffs2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vabsdiff2.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));






#line 3098 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsads2(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm("vabsdiff2.s32.s32.s32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));



#line 3111 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vabs4(unsigned int a)
{
    unsigned int r;

    unsigned int b = 0, c = 0;
    asm ("vabsdiff4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));











#line 3132 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vabsss4(unsigned int a)
{
    unsigned int r;

    unsigned int b = 0, c = 0;
    asm("vabsdiff4.s32.s32.s32.sat %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(c));














#line 3156 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vadd4(unsigned int a, unsigned int b)
{

    unsigned int r, c = 0;
    asm ("vadd4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));








#line 3173 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vaddss4 (unsigned int a, unsigned int b)
{

    unsigned int r, c = 0;
    asm ("vadd4.sat.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));

























































#line 3239 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vaddus4 (unsigned int a, unsigned int b)
{

    unsigned int r, c = 0;
    asm ("vadd4.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r):"r"(a),"r"(b),"r"(c));



















































#line 3299 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vavgs4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vavrg4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));


































#line 3343 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vavgu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vavrg4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));








#line 3361 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vhaddu4(unsigned int a, unsigned int b)
{
    
    
    unsigned int r, s;   
    s = a ^ b;           
    r = a & b;
    s = s & 0xfefefefe; 
    s = s >> 1;
    s = r + s;
    return s;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpeq4(unsigned int a, unsigned int b)
{
    unsigned int c, r;

    r = 0;
    asm ("vset4.u32.u32.eq %0,%1,%2,%3;" : "=r"(c) : "r"(a), "r"(b), "r"(r));
    r = c << 8;         
    r = r - c;          









#line 3395 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpges4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          

















#line 3424 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgeu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          




#line 3440 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgts4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          




















#line 3472 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpgtu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          




#line 3488 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmples4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          




















#line 3520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpleu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          




#line 3536 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmplts4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          



















#line 3567 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpltu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          




#line 3583 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vcmpne4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    c = r << 8;         
    r = c - r;          








#line 3603 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vabsdiff4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));






#line 3619 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vmaxs4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmax4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));





#line 3634 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vmaxu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmax4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));





#line 3649 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vmins4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmin4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));





#line 3664 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vminu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vmin4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));





#line 3679 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}
static __inline__ __attribute__((always_inline)) unsigned int __vseteq4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.eq %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));









#line 3697 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetles4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





















#line 3728 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetleu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.le %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 3743 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetlts4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));




















#line 3773 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetltu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.lt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 3788 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetges4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));


















#line 3816 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetgeu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.ge %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 3831 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetgts4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vset4.s32.s32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





















#line 3862 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetgtu4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.gt %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));





#line 3877 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsetne4(unsigned int a, unsigned int b)
{
    unsigned int r, c;

    c = 0;
    asm ("vset4.u32.u32.ne %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));









#line 3896 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsadu4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm("vabsdiff4.u32.u32.u32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));





#line 3911 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsub4(unsigned int a, unsigned int b)
{

    unsigned int r, c = 0;
    asm ("vsub4.u32.u32.u32 %0,%1,%2,%3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));








#line 3928 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsubss4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vsub4.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));

























































#line 3995 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsubus4(unsigned int a, unsigned int b)
{
    unsigned int r;

    unsigned int c = 0;
    asm ("vsub4.u32.u32.u32.sat %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(c));













































#line 4050 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vneg4(unsigned int a)
{
    return __vsub4 (0, a);
}

static __inline__ __attribute__((always_inline)) unsigned int __vnegss4(unsigned int a)
{
    unsigned int r;

    unsigned int s = 0;
    asm ("vsub4.s32.s32.s32.sat %0,%1,%2,%3;" : "=r"(r) :"r"(s),"r"(a),"r"(s));













#line 4078 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vabsdiffs4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm ("vabsdiff4.s32.s32.s32 %0,%1,%2,%3;" : "=r"(r) :"r"(a),"r"(b),"r"(s));






#line 4094 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}

static __inline__ __attribute__((always_inline)) unsigned int __vsads4(unsigned int a, unsigned int b)
{
    unsigned int r, s;

    s = 0;
    asm("vabsdiff4.s32.s32.s32.add %0,%1,%2,%3;":"=r"(r):"r"(a),"r"(b),"r"(s));





#line 4109 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
    return r;           
}






































































































































































































#line 4310 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"




#line 4315 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"




#line 4320 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"







#line 4328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"


#line 4512 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 4513 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"









































































































































#line 196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.hpp"



































































































































































#line 222 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.hpp"



#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.hpp"

#line 202 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
#line 203 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"

#line 205 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_atomic_functions.h"
#line 4515 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"


















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"




#line 1177 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"

#line 1179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"

#line 1181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"

static __inline__ __attribute__((always_inline)) double __dsub_rn(double a, double b);

static __inline__ __attribute__((always_inline)) double __dsub_rz(double a, double b);

static __inline__ __attribute__((always_inline)) double __dsub_ru(double a, double b);

static __inline__ __attribute__((always_inline)) double __dsub_rd(double a, double b);



#line 1193 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"


#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"





















































































































































































#line 183 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"




#line 188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"

#line 190 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"

#line 192 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"

static __inline__ __attribute__((always_inline)) double __dsub_rn(double a, double b)
{
  double res;
  asm ("sub.rn.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
  return res;
}

static __inline__ __attribute__((always_inline)) double __dsub_rz(double a, double b)
{
  double res;
  asm ("sub.rz.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
  return res;
}

static __inline__ __attribute__((always_inline)) double __dsub_ru(double a, double b)
{
  double res;
  asm ("sub.rp.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
  return res;
}

static __inline__ __attribute__((always_inline)) double __dsub_rd(double a, double b)
{
  double res;
  asm ("sub.rm.f64 %0, %1, %2;" : "=d"(res) : "d"(a), "d"(b));
  return res;
}



#line 224 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"

#line 226 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.hpp"

#line 1196 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"
#line 1197 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"

#line 1199 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_double_functions.h"

#line 4516 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"






































#line 97 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.hpp"























#line 82 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.hpp"



#line 86 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.hpp"

#line 103 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"
#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"

#line 106 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_atomic_functions.h"
#line 4517 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"




























































#line 119 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.hpp"

























































#line 116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.hpp"



#line 120 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.hpp"

#line 125 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"

#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
#line 4518 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_35_atomic_functions.h"























































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"






























































































































#line 128 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_atomic_functions.h"
#line 57 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_35_atomic_functions.h"

#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_35_atomic_functions.h"
#line 4519 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"























































#line 57 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"

#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"















































































































































































































































































































































































































































































#line 523 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.hpp"

























































































































































































































































































































































































































































#line 500 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.hpp"



#line 504 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.hpp"

#line 529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"
#line 530 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"

#line 532 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_60_atomic_functions.h"

#line 4520 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"






















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1489 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.hpp"


























































#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.hpp"



#line 121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.hpp"

#line 1495 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
#line 1496 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"
#line 1497 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_20_intrinsics.h"

#line 4521 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"

















































































































#line 172 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.hpp"






































































































































































































































































#line 321 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.hpp"



#line 325 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.hpp"

#line 178 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
#line 179 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"

#line 181 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_30_intrinsics.h"
#line 4522 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"




























































































































































































#line 247 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"




#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.hpp"


































































































































































































































































#line 317 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.hpp"



#line 321 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.hpp"


#line 252 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
#line 253 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"

#line 255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
#line 4523 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_35_intrinsics.h"














































































































#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"





























































































































































































































































#line 255 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_32_intrinsics.h"
#line 112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_35_intrinsics.h"



#line 116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_35_intrinsics.h"

#line 4524 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"


























































#line 117 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"





#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.hpp"




































































































#line 159 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.hpp"



#line 163 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.hpp"

#line 123 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
#line 124 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"

#line 126 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\sm_61_intrinsics.h"
#line 4525 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"

























































#line 59 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 2219 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"

extern uchar1     __surf1Dreadc1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf1Dreadc2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf1Dreadc4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf1Dreads1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf1Dreads2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf1Dreads4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf1Dreadu1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf1Dreadu2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf1Dreadu4(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf1Dreadl1(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf1Dreadl2(unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf2Dreadc1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf2Dreadc2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf2Dreadc4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf2Dreads1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf2Dreads2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf2Dreads4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf2Dreadu1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf2Dreadu2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf2Dreadu4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf2Dreadl1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf2Dreadl2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf3Dreadc1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf3Dreadc2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf3Dreadc4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf3Dreads1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf3Dreads2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf3Dreads4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf3Dreadu1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf3Dreadu2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf3Dreadu4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf3Dreadl1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf3Dreadl2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf1DLayeredreadc1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf1DLayeredreadc2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf1DLayeredreadc4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf1DLayeredreads1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf1DLayeredreads2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf1DLayeredreads4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf1DLayeredreadu1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf1DLayeredreadu2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf1DLayeredreadu4(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf1DLayeredreadl1(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf1DLayeredreadl2(unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern uchar1     __surf2DLayeredreadc1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar2     __surf2DLayeredreadc2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uchar4     __surf2DLayeredreadc4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort1    __surf2DLayeredreads1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort2    __surf2DLayeredreads2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ushort4    __surf2DLayeredreads4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint1      __surf2DLayeredreadu1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint2      __surf2DLayeredreadu2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern uint4      __surf2DLayeredreadu4(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong1 __surf2DLayeredreadl1(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern ulonglong2 __surf2DLayeredreadl2(unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec1(    uchar1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec2(    uchar2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritec4(    uchar4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites1(   ushort1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites2(   ushort2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwrites4(   ushort4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu1(     uint1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu2(     uint2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwriteu4(     uint4, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritel1(ulonglong1, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1Dwritel2(ulonglong2, unsigned long long, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec1(    uchar1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec2(    uchar2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritec4(    uchar4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites1(   ushort1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites2(   ushort2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwrites4(   ushort4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu1(     uint1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu2(     uint2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwriteu4(     uint4, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritel1(ulonglong1, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2Dwritel2(ulonglong2, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec1(    uchar1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec2(    uchar2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritec4(    uchar4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites1(   ushort1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites2(   ushort2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwrites4(   ushort4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu1(     uint1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu2(     uint2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwriteu4(     uint4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritel1(ulonglong1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf3Dwritel2(ulonglong2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec1(    uchar1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec2(    uchar2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritec4(    uchar4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites1(   ushort1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites2(   ushort2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwrites4(   ushort4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu1(     uint1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu2(     uint2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwriteu4(     uint4 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritel1(ulonglong1 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf1DLayeredwritel2(ulonglong2 val, unsigned long long, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec1(    uchar1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec2(    uchar2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritec4(    uchar4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites1(   ushort1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites2(   ushort2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwrites4(   ushort4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu1(     uint1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu2(     uint2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwriteu4(     uint4 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritel1(ulonglong1 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
extern void       __surf2DLayeredwritel2(ulonglong2 val, unsigned long long, int, int, int, enum cudaSurfaceBoundaryMode);
#line 2331 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"
#line 2332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_functions.h"

#line 4526 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"












































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 2766 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"

#line 2768 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_fetch_functions.h"


#line 4527 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"





































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1479 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"
#line 1480 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\texture_indirect_functions.h"

#line 4528 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"












































































































































































































































































































































































































































































































































































































































































































































































































































#line 814 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"

#line 816 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\surface_indirect_functions.h"



#line 4529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

#line 4531 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

#line 9846 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"



#line 9850 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 9852 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 9854 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"







static __inline__ __attribute__((always_inline)) float rintf(float a);

static __inline__ __attribute__((always_inline)) long int lrintf(float a);

static __inline__ __attribute__((always_inline)) long long int llrintf(float a);

static __inline__ __attribute__((always_inline)) float nearbyintf(float a);

static __inline__ __attribute__((always_inline)) int __signbitf(float a);



static __inline__ __attribute__((always_inline)) int __signbitl(double a);

static __inline__ __attribute__((always_inline)) int _ldsign(double a);

static __inline__ __attribute__((always_inline)) int __signbit(double a);
static __inline__ __attribute__((always_inline)) int _dsign(double a);


static __inline__ __attribute__((always_inline)) __inline__ __attribute__((always_inline)) int _fdsign(float a);

#line 9884 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

static __inline__ __attribute__((always_inline)) float copysignf(float a, float b);

static __inline__ __attribute__((always_inline)) int __finitef(float a);





#line 9894 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

static __inline__ __attribute__((always_inline)) int __isinff(float a);

static __inline__ __attribute__((always_inline)) int __isnanf(float a);

static __inline__ __attribute__((always_inline)) float nextafterf(float a, float b);

static __inline__ __attribute__((always_inline)) float nanf(const char *tagp);

static __inline__ __attribute__((always_inline)) float sinf(float a);

static __inline__ __attribute__((always_inline)) float cosf(float a);

static __inline__ __attribute__((always_inline)) void sincosf(float a, float *sptr, float *cptr);

static __inline__ __attribute__((always_inline)) float sinpif(float a);

static __inline__ __attribute__((always_inline)) float cospif(float a);

static __inline__ __attribute__((always_inline)) void sincospif(float a, float *sptr, float *cptr);

static __inline__ __attribute__((always_inline)) float tanf(float a);

static __inline__ __attribute__((always_inline)) float log2f(float a);

static __inline__ __attribute__((always_inline)) float expf(float a);

static __inline__ __attribute__((always_inline)) float exp10f(float a);

static __inline__ __attribute__((always_inline)) float coshf(float a);

static __inline__ __attribute__((always_inline)) float sinhf(float a);

static __inline__ __attribute__((always_inline)) float tanhf(float a);

static __inline__ __attribute__((always_inline)) float atan2f(float a, float b);

static __inline__ __attribute__((always_inline)) float atanf(float a);

static __inline__ __attribute__((always_inline)) float asinf(float a);

static __inline__ __attribute__((always_inline)) float acosf(float a);

static __inline__ __attribute__((always_inline)) float logf(float a);

static __inline__ __attribute__((always_inline)) float log10f(float a);

static __inline__ __attribute__((always_inline)) float log1pf(float a);

static __inline__ __attribute__((always_inline)) float acoshf(float a);

static __inline__ __attribute__((always_inline)) float asinhf(float a);

static __inline__ __attribute__((always_inline)) float atanhf(float a);

static __inline__ __attribute__((always_inline)) float expm1f(float a);

static __inline__ __attribute__((always_inline)) float hypotf(float a, float b);

static __inline__ __attribute__((always_inline)) float rhypotf(float a, float b) ;

static __inline__ __attribute__((always_inline)) float norm3df(float a, float b, float c) ;

static __inline__ __attribute__((always_inline)) float rnorm3df(float a, float b, float c) ;

static __inline__ __attribute__((always_inline)) float norm4df(float a, float b, float c, float d) ;

static __inline__ __attribute__((always_inline)) float cbrtf(float a);

static __inline__ __attribute__((always_inline)) float rcbrtf(float a);

static __inline__ __attribute__((always_inline)) float j0f(float a);

static __inline__ __attribute__((always_inline)) float j1f(float a);

static __inline__ __attribute__((always_inline)) float y0f(float a);

static __inline__ __attribute__((always_inline)) float y1f(float a);

static __inline__ __attribute__((always_inline)) float ynf(int n, float a);

static __inline__ __attribute__((always_inline)) float jnf(int n, float a);

static __inline__ __attribute__((always_inline)) float cyl_bessel_i0f(float a) ;

static __inline__ __attribute__((always_inline)) float cyl_bessel_i1f(float a) ;

static __inline__ __attribute__((always_inline)) float erff(float a);

static __inline__ __attribute__((always_inline)) float erfinvf(float a);

static __inline__ __attribute__((always_inline)) float erfcf(float a);

static __inline__ __attribute__((always_inline)) float erfcxf(float a);

static __inline__ __attribute__((always_inline)) float erfcinvf(float a);

static __inline__ __attribute__((always_inline)) float normcdfinvf(float a);

static __inline__ __attribute__((always_inline)) float normcdff(float a);

static __inline__ __attribute__((always_inline)) float lgammaf(float a);

static __inline__ __attribute__((always_inline)) float ldexpf(float a, int b);

static __inline__ __attribute__((always_inline)) float scalbnf(float a, int b);

static __inline__ __attribute__((always_inline)) float scalblnf(float a, long int b);

static __inline__ __attribute__((always_inline)) float frexpf(float a, int *b);

static __inline__ __attribute__((always_inline)) float modff(float a, float *b);

static __inline__ __attribute__((always_inline)) float fmodf(float a, float b);

static __inline__ __attribute__((always_inline)) float remainderf(float a, float b);

static __inline__ __attribute__((always_inline)) float remquof(float a, float b, int* quo);

static __inline__ __attribute__((always_inline)) float fmaf(float a, float b, float c);

static __inline__ __attribute__((always_inline)) float powif(float a, int b);

static __inline__ __attribute__((always_inline)) double powi(double a, int b);

static __inline__ __attribute__((always_inline)) float powf(float a, float b);

static __inline__ __attribute__((always_inline)) float tgammaf(float a);

static __inline__ __attribute__((always_inline)) float roundf(float a);

static __inline__ __attribute__((always_inline)) long long int llroundf(float a);

static __inline__ __attribute__((always_inline)) long int lroundf(float a);

static __inline__ __attribute__((always_inline)) float fdimf(float a, float b);

static __inline__ __attribute__((always_inline)) int ilogbf(float a);

static __inline__ __attribute__((always_inline)) float logbf(float a);































































































































































































































































































#line 10322 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 10324 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"



#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"


































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 1092 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"


#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
















































































































































































































































#line 242 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\host_defines.h"
#line 1095 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_constants.h"























































































































































#line 153 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_constants.h"
#line 1096 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"








#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"




























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 12542 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions_decls.h"

#line 1105 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"

#line 1107 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

















































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 4531 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.h"

#line 1108 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"



#line 1112 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"

#line 1114 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"

#line 1116 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"




#line 1121 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"














static __inline__ __attribute__((always_inline)) float rintf(float a) 
{
  return __nv_rintf(a);
}

static __inline__ __attribute__((always_inline)) long int lrintf(float a) 
{


#line 1145 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
  return (long int)__float2int_rn(a);
#line 1147 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
}

static __inline__ __attribute__((always_inline)) long long int llrintf(float a)  
{
  return __nv_llrintf(a);
}

static __inline__ __attribute__((always_inline)) float nearbyintf(float a)  
{
  return __nv_nearbyintf(a);
}







static __inline__ __attribute__((always_inline)) int __signbitf(float a) 
{
  return __nv_signbitf(a);
}


static __inline__ __attribute__((always_inline)) int __signbitl(double a);
static __inline__ __attribute__((always_inline)) int _ldsign(double a)
{
  return __signbitl(a);
}

static __inline__ __attribute__((always_inline)) int __signbit(double a);
static __inline__ __attribute__((always_inline)) int _dsign(double a)
{
  return __signbit(a);
}

static __inline__ __attribute__((always_inline)) int _fdsign(float a)
{
  return __signbitf(a);
}
#line 1188 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"

static __inline__ __attribute__((always_inline)) float copysignf(float a, float b) 
{
  return __nv_copysignf(a, b);
}

static __inline__ __attribute__((always_inline)) int __finitef(float a) 
{
  return __nv_finitef(a);
}








#line 1207 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"

static __inline__ __attribute__((always_inline)) int __isinff(float a) 
{
  return __nv_isinff(a);
}

static __inline__ __attribute__((always_inline)) int __isnanf(float a) 
{
  return __nv_isnanf(a);
}

static __inline__ __attribute__((always_inline)) float nextafterf(float a, float b) 
{
  return __nv_nextafterf(a, b);
}

static __inline__ __attribute__((always_inline)) float nanf(const char *tagp) 
{
  return __nv_nanf((const signed char *) tagp);
}







static __inline__ __attribute__((always_inline)) float sinf(float a) 
{
  if (1) {
    return __nv_fast_sinf(a);
  } else {
    return __nv_sinf(a);
  }
}

static __inline__ __attribute__((always_inline)) float cosf(float a) 
{
  if (1) {
    return __nv_fast_cosf(a);
  } else {
    return __nv_cosf(a);
  }
}

static __inline__ __attribute__((always_inline)) void sincosf(float a, float *sptr, float *cptr) 
{
  if (1) {
    __nv_fast_sincosf(a, sptr, cptr);
  } else {
    __nv_sincosf(a, sptr, cptr);
  }
}

static __inline__ __attribute__((always_inline)) float sinpif(float a) 
{
  return __nv_sinpif(a);
}

static __inline__ __attribute__((always_inline)) float cospif(float a) 
{
  return __nv_cospif(a);
}

static __inline__ __attribute__((always_inline)) void sincospif(float a, float *sptr, float *cptr)  
{
  __nv_sincospif(a, sptr, cptr);
}

static __inline__ __attribute__((always_inline)) float tanf(float a) 
{
  if (1) {
    return __nv_fast_tanf(a);
  } else {
    return __nv_tanf(a);
  }
}

static __inline__ __attribute__((always_inline)) float log2f(float a) 
{
  if (1) {
    return __nv_fast_log2f(a);
  } else {
    return __nv_log2f(a);
  }
}

static __inline__ __attribute__((always_inline)) float expf(float a) 
{
  if (1) {
    return __nv_fast_expf(a);
  } else {
    return __nv_expf(a);
  }
}

static __inline__ __attribute__((always_inline)) float exp10f(float a) 
{
  if (1) {
    return __nv_fast_exp10f(a);
  } else {
    return __nv_exp10f(a);
  }
}

static __inline__ __attribute__((always_inline)) float coshf(float a) 
{
  return __nv_coshf(a);
}

static __inline__ __attribute__((always_inline)) float sinhf(float a) 
{
  return __nv_sinhf(a);
}

static __inline__ __attribute__((always_inline)) float tanhf(float a) 
{
  return __nv_tanhf(a);
}

static __inline__ __attribute__((always_inline)) float atan2f(float a, float b) 
{
  return __nv_atan2f(a, b);
}

static __inline__ __attribute__((always_inline)) float atanf(float a) 
{
  return __nv_atanf(a);
}

static __inline__ __attribute__((always_inline)) float asinf(float a) 
{
  return __nv_asinf(a);
}

static __inline__ __attribute__((always_inline)) float acosf(float a) 
{
  return __nv_acosf(a);
}

static __inline__ __attribute__((always_inline)) float logf(float a) 
{
  if (1) {
    return __nv_fast_logf(a);
  } else {
    return __nv_logf(a);
  }
}

static __inline__ __attribute__((always_inline)) float log10f(float a) 
{
  if (1) {
    return __nv_fast_log10f(a);
  } else {
    return __nv_log10f(a);
  }
}

static __inline__ __attribute__((always_inline)) float log1pf(float a) 
{
  return __nv_log1pf(a);
}

static __inline__ __attribute__((always_inline)) float acoshf(float a) 
{
  return __nv_acoshf(a);
}

static __inline__ __attribute__((always_inline)) float asinhf(float a) 
{
  return __nv_asinhf(a);
}

static __inline__ __attribute__((always_inline)) float atanhf(float a) 
{
  return __nv_atanhf(a);
}

static __inline__ __attribute__((always_inline)) float expm1f(float a) 
{
  return __nv_expm1f(a);
}

static __inline__ __attribute__((always_inline)) float hypotf(float a, float b) 
{
  return __nv_hypotf(a, b);
}

static __inline__ __attribute__((always_inline)) float rhypotf(float a, float b) 
{
  return __nv_rhypotf(a, b);
}

static __inline__ __attribute__((always_inline)) float rnormf(int dim, float const * a) 
{
  return __nv_rnormf(dim, a);
}

static __inline__ __attribute__((always_inline)) float normf(int dim, float const * a) 
{
  return __nv_normf(dim, a);
}

static __inline__ __attribute__((always_inline)) float norm3df(float a, float b, float c) 
{
  return __nv_norm3df(a, b, c);
}

static __inline__ __attribute__((always_inline)) float rnorm3df(float a, float b, float c) 
{
  return __nv_rnorm3df(a, b, c);
}

static __inline__ __attribute__((always_inline)) float norm4df(float a, float b, float c, float d) 
{
  return __nv_norm4df(a, b, c, d);
}

static __inline__ __attribute__((always_inline)) float rnorm4df(float a, float b, float c, float d) 
{
  return __nv_rnorm4df(a, b, c, d);
}

static __inline__ __attribute__((always_inline)) float cbrtf(float a) 
{
  return __nv_cbrtf(a);
}

static __inline__ __attribute__((always_inline)) float rcbrtf(float a) 
{
  return __nv_rcbrtf(a);
}

static __inline__ __attribute__((always_inline)) float j0f(float a) 
{
  return __nv_j0f(a);
}

static __inline__ __attribute__((always_inline)) float j1f(float a) 
{
  return __nv_j1f(a);
}

static __inline__ __attribute__((always_inline)) float y0f(float a) 
{
  return __nv_y0f(a);
}

static __inline__ __attribute__((always_inline)) float y1f(float a) 
{
  return __nv_y1f(a);
}

static __inline__ __attribute__((always_inline)) float ynf(int n, float a) 
{
  return __nv_ynf(n, a);
}

static __inline__ __attribute__((always_inline)) float jnf(int n, float a) 
{
  return __nv_jnf(n, a);
}

static __inline__ __attribute__((always_inline)) float cyl_bessel_i0f(float a) 
{
  return __nv_cyl_bessel_i0f(a);
}

static __inline__ __attribute__((always_inline)) float cyl_bessel_i1f(float a) 
{
  return __nv_cyl_bessel_i1f(a);
}

static __inline__ __attribute__((always_inline)) float erff(float a) 
{
  return __nv_erff(a);
}

static __inline__ __attribute__((always_inline)) float erfinvf(float a) 
{
  return __nv_erfinvf(a);
}

static __inline__ __attribute__((always_inline)) float erfcf(float a) 
{
  return __nv_erfcf(a);
}

static __inline__ __attribute__((always_inline)) float erfcxf(float a) 
{
  return __nv_erfcxf(a);
}

static __inline__ __attribute__((always_inline)) float erfcinvf(float a) 
{
  return __nv_erfcinvf(a);
}

static __inline__ __attribute__((always_inline)) float normcdfinvf(float a) 
{
  return __nv_normcdfinvf(a);
}

static __inline__ __attribute__((always_inline)) float normcdff(float a) 
{
  return __nv_normcdff(a);
}

static __inline__ __attribute__((always_inline)) float lgammaf(float a) 
{
  return __nv_lgammaf(a);
}

static __inline__ __attribute__((always_inline)) float ldexpf(float a, int b) 
{
  return __nv_ldexpf(a, b);
}

static __inline__ __attribute__((always_inline)) float scalbnf(float a, int b) 
{
  return __nv_scalbnf(a, b);
}

static __inline__ __attribute__((always_inline)) float scalblnf(float a, long int b) 
{
  int t;
  if (b > 2147483647L) {
    t = 2147483647;
  } else if (b < (-2147483647 - 1)) {
    t = (-2147483647 - 1);
  } else {
    t = (int)b;
  }
  return scalbnf(a, t);
}

static __inline__ __attribute__((always_inline)) float frexpf(float a, int *b) 
{
  return __nv_frexpf(a, b);
}

static __inline__ __attribute__((always_inline)) float modff(float a, float *b) 
{
  return __nv_modff(a, b);
}

static __inline__ __attribute__((always_inline)) float fmodf(float a, float b) 
{
  return __nv_fmodf(a, b);
}

static __inline__ __attribute__((always_inline)) float remainderf(float a, float b) 
{
  return __nv_remainderf(a, b);
}

static __inline__ __attribute__((always_inline)) float remquof(float a, float b, int* quo) 
{
  return __nv_remquof(a, b, quo);
}

static __inline__ __attribute__((always_inline)) float fmaf(float a, float b, float c) 
{
  return __nv_fmaf(a, b, c);
}

static __inline__ __attribute__((always_inline)) float powif(float a, int b) 
{
  return __nv_powif(a, b);
}

static __inline__ __attribute__((always_inline)) double powi(double a, int b) 
{
  return __nv_powi(a, b);
}

static __inline__ __attribute__((always_inline)) float powf(float a, float b) 
{
  if (1) {
    return __nv_fast_powf(a, b);
  } else {
    return __nv_powf(a, b);
  }
}

static __inline__ __attribute__((always_inline)) float tgammaf(float a) 
{
  return __nv_tgammaf(a);
}

static __inline__ __attribute__((always_inline)) float roundf(float a) 
{
  return __nv_roundf(a);
}

static __inline__ __attribute__((always_inline)) long long int llroundf(float a) 
{
  return __nv_llroundf(a);
}

static __inline__ __attribute__((always_inline)) long int lroundf(float a) 
{


#line 1612 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
  return (long int)(roundf(a));
#line 1614 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"
}

static __inline__ __attribute__((always_inline)) float fdimf(float a, float b) 
{
  return __nv_fdimf(a, b);
}

static __inline__ __attribute__((always_inline)) int ilogbf(float a) 
{
  return __nv_ilogbf(a);
}

static __inline__ __attribute__((always_inline)) float logbf(float a) 
{
  return __nv_logbf(a);
}














































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































#line 3853 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"

#line 3855 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"

#line 3857 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.hpp"


#line 10328 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 10330 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"











static __inline__ __attribute__((always_inline)) double rint(double a) ; 

static __inline__ __attribute__((always_inline)) long int lrint(double a) ;

static __inline__ __attribute__((always_inline)) long long int llrint(double a) ;

static __inline__ __attribute__((always_inline)) double nearbyint(double a) ;







static __inline__ __attribute__((always_inline)) int __signbitd(double a);

static __inline__ __attribute__((always_inline)) int __isfinited(double a);

static __inline__ __attribute__((always_inline)) int __isinfd(double a);

static __inline__ __attribute__((always_inline)) int __isnand(double a);











#line 104 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"

static __inline__ __attribute__((always_inline)) int __signbit(double a) ;

static __inline__ __attribute__((always_inline)) int __signbitl(double a);

static __inline__ __attribute__((always_inline)) int __finite(double a) ; 

static __inline__ __attribute__((always_inline)) int __finitel(double a);

static __inline__ __attribute__((always_inline)) int __isinf(double a) ;

static __inline__ __attribute__((always_inline)) int __isinfl(double a);

static __inline__ __attribute__((always_inline)) int __isnan(double a) ;

static __inline__ __attribute__((always_inline)) int __isnanl(double a);

#line 122 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"

static __inline__ __attribute__((always_inline)) double copysign(double a, double b) ;

static __inline__ __attribute__((always_inline)) void sincos(double a, double *sptr, double *cptr) ;

static __inline__ __attribute__((always_inline)) void sincospi(double a, double *sptr, double *cptr);

static __inline__ __attribute__((always_inline)) double sin(double a) ;

static __inline__ __attribute__((always_inline)) double cos(double a) ;

static __inline__ __attribute__((always_inline)) double sinpi(double a);

static __inline__ __attribute__((always_inline)) double cospi(double a);

static __inline__ __attribute__((always_inline)) double tan(double a) ;

static __inline__ __attribute__((always_inline)) double log(double a) ;

static __inline__ __attribute__((always_inline)) double log2(double a) ;

static __inline__ __attribute__((always_inline)) double log10(double a) ;

static __inline__ __attribute__((always_inline)) double log1p(double a) ;

static __inline__ __attribute__((always_inline)) double exp(double a) ;

static __inline__ __attribute__((always_inline)) double exp2(double a) ;

static __inline__ __attribute__((always_inline)) double exp10(double a) ;

static __inline__ __attribute__((always_inline)) double expm1(double a) ;

static __inline__ __attribute__((always_inline)) double cosh(double a) ;

static __inline__ __attribute__((always_inline)) double sinh(double a) ;

static __inline__ __attribute__((always_inline)) double tanh(double a) ;

static __inline__ __attribute__((always_inline)) double atan2(double a, double b) ;

static __inline__ __attribute__((always_inline)) double atan(double a) ;

static __inline__ __attribute__((always_inline)) double asin(double a) ;

static __inline__ __attribute__((always_inline)) double acos(double a) ;

static __inline__ __attribute__((always_inline)) double acosh(double a) ;

static __inline__ __attribute__((always_inline)) double asinh(double a) ;

static __inline__ __attribute__((always_inline)) double atanh(double a) ;

static __inline__ __attribute__((always_inline)) double hypot(double a, double b) ;

static __inline__ __attribute__((always_inline)) double rhypot(double a, double b) ;

static __inline__ __attribute__((always_inline)) double norm3d(double a, double b, double c) ;

static __inline__ __attribute__((always_inline)) double rnorm3d(double a, double b, double c) ;

static __inline__ __attribute__((always_inline)) double norm4d(double a, double b, double c, double d) ;

static __inline__ __attribute__((always_inline)) double rnorm4d(double a, double b, double c, double d) ;

static __inline__ __attribute__((always_inline)) double norm(int dim, double const * t) ;

static __inline__ __attribute__((always_inline)) double rnorm(int dim, double const * t) ;

static __inline__ __attribute__((always_inline)) double cbrt(double a) ;

static __inline__ __attribute__((always_inline)) double rcbrt(double a);

static __inline__ __attribute__((always_inline)) double pow(double a, double b) ;

static __inline__ __attribute__((always_inline)) double j0(double a) ;

static __inline__ __attribute__((always_inline)) double j1(double a) ;

static __inline__ __attribute__((always_inline)) double y0(double a) ;

static __inline__ __attribute__((always_inline)) double y1(double a) ;

static __inline__ __attribute__((always_inline)) double yn(int n, double a) ;

static __inline__ __attribute__((always_inline)) double jn(int n, double a) ;

static __inline__ __attribute__((always_inline)) double cyl_bessel_i0(double a) ;

static __inline__ __attribute__((always_inline)) double cyl_bessel_i1(double a) ;

static __inline__ __attribute__((always_inline)) double erf(double a) ;

static __inline__ __attribute__((always_inline)) double erfinv(double a);

static __inline__ __attribute__((always_inline)) double erfcinv(double a);

static __inline__ __attribute__((always_inline)) double normcdfinv(double a);

static __inline__ __attribute__((always_inline)) double erfc(double a)  ;

static __inline__ __attribute__((always_inline)) double erfcx(double a);

static __inline__ __attribute__((always_inline)) double normcdf(double a);

static __inline__ __attribute__((always_inline)) double tgamma(double a) ;

static __inline__ __attribute__((always_inline)) double lgamma(double a) ;

static __inline__ __attribute__((always_inline)) double ldexp(double a, int b) ;

static __inline__ __attribute__((always_inline)) double scalbn(double a, int b) ;

static __inline__ __attribute__((always_inline)) double scalbln(double a, long int b) ;

static __inline__ __attribute__((always_inline)) double frexp(double a, int *b) ;

static __inline__ __attribute__((always_inline)) double modf(double a, double *b) ;

static __inline__ __attribute__((always_inline)) double fmod(double a, double b) ;

static __inline__ __attribute__((always_inline)) double remainder(double a, double b) ;

static __inline__ __attribute__((always_inline)) double remquo(double a, double b, int *c) ;

static __inline__ __attribute__((always_inline)) double nextafter(double a, double b) ;

static __inline__ __attribute__((always_inline)) double nan(const char *tagp) ;

static __inline__ __attribute__((always_inline)) double round(double a) ;

static __inline__ __attribute__((always_inline)) long long int llround(double a) ;

static __inline__ __attribute__((always_inline)) long int lround(double a) ;

static __inline__ __attribute__((always_inline)) double fdim(double a, double b) ;

static __inline__ __attribute__((always_inline)) int ilogb(double a) ;

static __inline__ __attribute__((always_inline)) double logb(double a) ;

static __inline__ __attribute__((always_inline)) double fma(double a, double b, double c) ;

#line 266 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"




#line 1 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"






















































#line 56 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"

#line 58 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"

#line 60 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"











static __inline__ __attribute__((always_inline)) double rint(double a) 
{
  return __nv_rint(a);
}

static __inline__ __attribute__((always_inline)) long int lrint(double a) 
{


#line 81 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"
  return (long int)__double2int_rn(a);
#line 83 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"
}

static __inline__ __attribute__((always_inline)) long long int llrint(double a) 
{
  return __nv_llrint(a);
}

static __inline__ __attribute__((always_inline)) double nearbyint(double a) 
{
  return __nv_nearbyint(a);
}







static __inline__ __attribute__((always_inline)) int __signbitd(double a)
{
  return __nv_signbitd(a);
}

static __inline__ __attribute__((always_inline)) int __isfinited(double a)
{
  return __nv_isfinited(a);
}

static __inline__ __attribute__((always_inline)) int __isinfd(double a)
{
  return __nv_isinfd(a);
}

static __inline__ __attribute__((always_inline)) int __isnand(double a)
{
  return __nv_isnand(a);
}























#line 144 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"

static __inline__ __attribute__((always_inline)) int __signbit(double a) 
{
  return __signbitd(a);
}

static __inline__ __attribute__((always_inline)) int __signbitl(double a)
{
  return __signbit((double)a);
}

static __inline__ __attribute__((always_inline)) int __finite(double a) 
{
  return __isfinited(a);
}

static __inline__ __attribute__((always_inline)) int __finitel(double a)
{
  return __finite((double)a);
}

static __inline__ __attribute__((always_inline)) int __isinf(double a) 
{
  return __isinfd(a);
}

static __inline__ __attribute__((always_inline)) int __isinfl(double a)
{
  return __isinf((double)a);
}

static __inline__ __attribute__((always_inline)) int __isnan(double a) 
{
  return __isnand(a);
}

static __inline__ __attribute__((always_inline)) int __isnanl(double a)
{
  return __isnan((double)a);
}

#line 186 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"

static __inline__ __attribute__((always_inline)) double copysign(double a, double b) 
{
  return __nv_copysign(a, b);
}

static __inline__ __attribute__((always_inline)) void sincos(double a, double *sptr, double *cptr) 
{
  __nv_sincos(a, sptr, cptr);
}

static __inline__ __attribute__((always_inline)) void sincospi(double a, double *sptr, double *cptr)
{
  __nv_sincospi(a, sptr, cptr);
}

static __inline__ __attribute__((always_inline)) double sin(double a) 
{
  return __nv_sin(a);
}

static __inline__ __attribute__((always_inline)) double cos(double a) 
{
  return __nv_cos(a);
}

static __inline__ __attribute__((always_inline)) double sinpi(double a)
{
  return __nv_sinpi(a);
}

static __inline__ __attribute__((always_inline)) double cospi(double a)
{
  return __nv_cospi(a);
}

static __inline__ __attribute__((always_inline)) double tan(double a) 
{
  return __nv_tan(a);
}

static __inline__ __attribute__((always_inline)) double log(double a) 
{
  return __nv_log(a);
}

static __inline__ __attribute__((always_inline)) double log2(double a) 
{
  return __nv_log2(a);
}

static __inline__ __attribute__((always_inline)) double log10(double a) 
{
  return __nv_log10(a);
}

static __inline__ __attribute__((always_inline)) double log1p(double a) 
{
  return __nv_log1p(a);
}

static __inline__ __attribute__((always_inline)) double exp(double a) 
{
  return __nv_exp(a);
}

static __inline__ __attribute__((always_inline)) double exp2(double a) 
{
  return __nv_exp2(a);
}

static __inline__ __attribute__((always_inline)) double exp10(double a) 
{
  return __nv_exp10(a);
}

static __inline__ __attribute__((always_inline)) double expm1(double a) 
{
  return __nv_expm1(a);
}

static __inline__ __attribute__((always_inline)) double cosh(double a) 
{
  return __nv_cosh(a);
}

static __inline__ __attribute__((always_inline)) double sinh(double a) 
{
  return __nv_sinh(a);
}

static __inline__ __attribute__((always_inline)) double tanh(double a) 
{
  return __nv_tanh(a);
}

static __inline__ __attribute__((always_inline)) double atan2(double a, double b) 
{
  return __nv_atan2(a, b);
}

static __inline__ __attribute__((always_inline)) double atan(double a) 
{
  return __nv_atan(a);
}

static __inline__ __attribute__((always_inline)) double asin(double a) 
{
  return __nv_asin(a);
}

static __inline__ __attribute__((always_inline)) double acos(double a) 
{
  return __nv_acos(a);
}

static __inline__ __attribute__((always_inline)) double acosh(double a) 
{
  return __nv_acosh(a);
}

static __inline__ __attribute__((always_inline)) double asinh(double a) 
{
  return __nv_asinh(a);  
}

static __inline__ __attribute__((always_inline)) double atanh(double a) 
{
  return __nv_atanh(a);
}

static __inline__ __attribute__((always_inline)) double hypot(double a, double b) 
{
  return __nv_hypot(a, b);
}

static __inline__ __attribute__((always_inline)) double rhypot(double a, double b) 
{
  return __nv_rhypot(a, b);
}

static __inline__ __attribute__((always_inline)) double norm3d(double a, double b, double c) 
{
  return __nv_norm3d(a, b, c);
}

static __inline__ __attribute__((always_inline)) double rnorm3d(double a, double b, double c) 
{
  return __nv_rnorm3d(a, b, c);
}

static __inline__ __attribute__((always_inline)) double norm4d(double a, double b, double c, double d) 
{
  return __nv_norm4d(a, b, c, d);
}

static __inline__ __attribute__((always_inline)) double rnorm4d(double a, double b, double c, double d) 
{
  return __nv_rnorm4d(a, b, c, d);
}

static __inline__ __attribute__((always_inline)) double norm(int dim, double const * t) 
{
  return __nv_norm(dim, t);
}

static __inline__ __attribute__((always_inline)) double rnorm(int dim, double const * t) 
{
  return __nv_rnorm(dim, t);
}

static __inline__ __attribute__((always_inline)) double cbrt(double a) 
{
  return __nv_cbrt(a);
}

static __inline__ __attribute__((always_inline)) double rcbrt(double a)
{
  return __nv_rcbrt(a);
}

static __inline__ __attribute__((always_inline)) double pow(double a, double b) 
{
  return __nv_pow(a, b);
}

static __inline__ __attribute__((always_inline)) double j0(double a) 
{
  return __nv_j0(a);
}

static __inline__ __attribute__((always_inline)) double j1(double a) 
{
  return __nv_j1(a);
}

static __inline__ __attribute__((always_inline)) double y0(double a) 
{
  return __nv_y0(a);
}

static __inline__ __attribute__((always_inline)) double y1(double a) 
{
  return __nv_y1(a);
}

static __inline__ __attribute__((always_inline)) double yn(int n, double a) 
{
  return __nv_yn(n, a);
}

static __inline__ __attribute__((always_inline)) double jn(int n, double a) 
{
  return __nv_jn(n, a);
}

static __inline__ __attribute__((always_inline)) double cyl_bessel_i0(double a) 
{
  return __nv_cyl_bessel_i0(a);
}

static __inline__ __attribute__((always_inline)) double cyl_bessel_i1(double a) 
{
  return __nv_cyl_bessel_i1(a);
}

static __inline__ __attribute__((always_inline)) double erf(double a) 
{
  return __nv_erf(a);
}

static __inline__ __attribute__((always_inline)) double erfinv(double a)
{
  return __nv_erfinv(a);
}

static __inline__ __attribute__((always_inline)) double erfcinv(double a)
{
  return __nv_erfcinv(a);
}

static __inline__ __attribute__((always_inline)) double normcdfinv(double a)
{
  return __nv_normcdfinv(a);
}

static __inline__ __attribute__((always_inline)) double erfc(double a)    
{  
  return __nv_erfc(a);
}

static __inline__ __attribute__((always_inline)) double erfcx(double a)  
{
  return __nv_erfcx(a);
}

static __inline__ __attribute__((always_inline)) double normcdf(double a)
{
  return __nv_normcdf(a);
}

static __inline__ __attribute__((always_inline)) double tgamma(double a) 
{
  return __nv_tgamma(a);
}

static __inline__ __attribute__((always_inline)) double lgamma(double a) 
{
  return __nv_lgamma(a);
}

static __inline__ __attribute__((always_inline)) double ldexp(double a, int b) 
{
  return __nv_ldexp(a, b);
}

static __inline__ __attribute__((always_inline)) double scalbn(double a, int b) 
{
  return __nv_scalbn(a, b);
}

static __inline__ __attribute__((always_inline)) double scalbln(double a, long int b) 
{




#line 474 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"
  return scalbn(a, (int)b);
}

static __inline__ __attribute__((always_inline)) double frexp(double a, int *b) 
{
  return __nv_frexp(a, b);  
}

static __inline__ __attribute__((always_inline)) double modf(double a, double *b) 
{
  return __nv_modf(a, b);
}

static __inline__ __attribute__((always_inline)) double fmod(double a, double b) 
{
  return __nv_fmod(a, b);
}

static __inline__ __attribute__((always_inline)) double remainder(double a, double b) 
{
  return __nv_remainder(a, b);
}

static __inline__ __attribute__((always_inline)) double remquo(double a, double b, int *c) 
{
  return __nv_remquo(a, b, c);
}

static __inline__ __attribute__((always_inline)) double nextafter(double a, double b) 
{
  return __nv_nextafter(a, b);
}

static __inline__ __attribute__((always_inline)) double nan(const char *tagp) 
{
  return __nv_nan((const signed char *) tagp);
}

static __inline__ __attribute__((always_inline)) double round(double a) 
{
  return __nv_round(a);
}

static __inline__ __attribute__((always_inline)) long long int llround(double a) 
{
  return __nv_llround(a);
}

static __inline__ __attribute__((always_inline)) long int lround(double a) 
{


#line 527 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"
  return (long int)round(a);
#line 529 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"
}

static __inline__ __attribute__((always_inline)) double fdim(double a, double b) 
{
  return __nv_fdim(a, b);
}

static __inline__ __attribute__((always_inline)) int ilogb(double a) 
{
  return __nv_ilogb(a);
}

static __inline__ __attribute__((always_inline)) double logb(double a) 
{
  return __nv_logb(a);
}

static __inline__ __attribute__((always_inline)) double fma(double a, double b, double c)  
{
  return __nv_fma(a, b, c);
}

#line 552 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"



#line 556 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.hpp"

#line 271 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"
#line 272 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"

#line 274 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions_dbl_ptx3.h"
#line 10332 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 10334 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\math_functions.h"

#line 250 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\common_functions.h"

#line 252 "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include\\common_functions.h"

#line 27 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"

#line 29 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"

#line 31 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"

#line 33 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"

#line 35 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"

#line 37 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"

#line 39 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
#line 106 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
 __attribute__((nv_linkonce_odr))  __inline__ float _Z3expf(
#line 106 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
float _Xx){
#line 106 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
{
#line 107 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
{
#line 108 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
return expf(_Xx);
#line 109 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
}
#line 109 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
}}
#line 121 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
 __attribute__((nv_linkonce_odr))  __inline__ float _Z4fabsf(
#line 121 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
float _Xx){
#line 121 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
{
#line 122 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
{
#line 123 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
return fabsf(_Xx);
#line 124 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
}
#line 124 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
}}
#line 248 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
 __attribute__((nv_linkonce_odr))  __inline__ float _Z3powff(
#line 248 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
float _Xx, 
#line 249 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
float _Yx){
#line 249 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
{
#line 250 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
{
#line 251 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
return powf(_Xx, _Yx);
#line 252 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
}
#line 252 "C:\\Program Files (x86)\\Microsoft Visual Studio 14.0\\VC\\include\\cmath"
}}
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
static  __inline__ void _ZN39_INTERNAL_17_CudaStuff_cpp1_ii_1abe6ff811syncthreadsEv(void){
#line 168 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
{
#line 169 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
{
#line 170 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
__syncthreads();
#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
} 
#line 171 "c:\\program files\\nvidia gpu computing toolkit\\cuda\\v8.0\\include\\device_functions.hpp"
}}
#line 82 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float _Z9Cuefun_caf(
#line 82 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float z){
#line 82 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 82 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 83 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
if (((double)(_Z4fabsf(z))) < (0.0001))
#line 83 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 83 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 84 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return (1.0F) - ((float)(fdividef(((double)z), (2.0))));
#line 85 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 85 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
else 
#line 85 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 85 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 86 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return (float)(fdividef(((double)z), ((double)((_Z3expf(z)) - (1.0F)))));
#line 87 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 87 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 88 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 88 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 89 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float _Z9Cuefun_kmf(
#line 89 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float z){
#line 89 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 89 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 90 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
if (((double)(_Z4fabsf(z))) < (0.0001))
#line 90 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 90 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 91 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return (1.0F) - ((float)(fdividef(((double)z), (2.0))));
#line 92 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 92 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
else 
#line 92 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 92 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 93 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return (float)(fdividef(((double)z), ((double)((_Z3expf(z)) - (1.0F)))));
#line 94 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 94 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 95 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 95 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 96 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float _Z9Cuefun_kvf(
#line 96 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float z){
#line 96 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 96 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 97 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
if (((double)(_Z4fabsf(z))) < (0.0001))
#line 97 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 97 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 98 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return (1.0F) - ((float)(fdividef(((double)z), (2.0))));
#line 99 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 99 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
else 
#line 99 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 99 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 100 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return (float)(fdividef(((double)z), ((double)((_Z3expf(z)) - (1.0F)))));
#line 101 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 101 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 102 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 102 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float _Z10Cutrap0_naffff(
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float th, 
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float a, 
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float q){
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 103 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 104 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
if (((double)(_Z4fabsf(((float)(fdividef(((double)(v - th)), ((double)q))))))) > (9.9999999999999995e-007))
#line 104 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 104 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 105 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return (float)(fdividef(((double)(a * (v - th))), ((double)((1.0F) - (_Z3expf(((float)(fdividef(((double)(-(v - th))), ((double)q))))))))));
#line 106 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 106 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
else 
#line 106 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 106 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 107 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
return a * q;
#line 108 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 108 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 109 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 109 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z11Cutrates_cafffRfS_S_S_(
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_ca, 
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cao_ca, 
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *hinf, 
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *htau, 
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *minf, 
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *mtau){
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 112 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 114 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z10Curates_cafffRfS_S_S_(v, gbar_ca, cao_ca, hinf, htau, minf, mtau);
#line 115 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 115 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z10Curates_cafffRfS_S_S_(
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float vm, 
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_ca, 
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cao_ca, 
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *hinf, 
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *htau, 
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *minf, 
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *mtau){
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 116 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 117 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242113_8_non_const_a;
#line 117 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242113_11_non_const_b;
#line 119 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242113_8_non_const_a = ((float)((0.20899999999999999) * ((double)(_Z9Cuefun_caf(((float)(fdivide(((double)(-((27.0F) + vm))), (3.7999999999999998)))))))));
#line 120 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242113_11_non_const_b = ((float)((0.93999999999999995) * ((double)(_Z3expf(((float)(fdividef(((double)((-75.0F) - vm)), (17.0)))))))));
#line 121 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 122 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*mtau) = ((float)(fdivide((0.31158821952278315), ((double)(__cuda_local_var_242113_8_non_const_a + __cuda_local_var_242113_11_non_const_b)))));
#line 123 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*minf) = ((float)(fdividef(((double)__cuda_local_var_242113_8_non_const_a), ((double)(__cuda_local_var_242113_8_non_const_a + __cuda_local_var_242113_11_non_const_b)))));
#line 124 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 125 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242113_8_non_const_a = ((float)((0.000457) * ((double)(_Z3expf(((float)(fdividef(((double)((-13.0F) - vm)), (50.0)))))))));
#line 126 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242113_11_non_const_b = ((float)(fdivide((0.0064999999999999997), ((double)((_Z3expf(((float)(fdividef(((double)((-vm) - (15.0F))), (28.0)))))) + (1.0F))))));
#line 127 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*htau) = ((float)(fdivide((0.31158821952278315), ((double)(__cuda_local_var_242113_8_non_const_a + __cuda_local_var_242113_11_non_const_b)))));
#line 128 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*hinf) = ((float)(fdividef(((double)__cuda_local_var_242113_8_non_const_a), ((double)(__cuda_local_var_242113_8_non_const_a + __cuda_local_var_242113_11_non_const_b)))));
#line 129 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 129 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z11Curates_kcafffffRfS_S_S_(
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cai, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kca, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float caix_kca, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kca, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kca, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *a, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *b, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ninf, 
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ntau){
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 130 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 132 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 133 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*a) = (_Z3powff((((float)Ra_kca) * cai), ((float)caix_kca)));
#line 134 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*b) = Rb_kca;
#line 136 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ntau) = ((float)(fdivide((0.31158821952278315), ((double)((*a) + (*b))))));
#line 137 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ninf) = ((float)(fdividef(((double)(*a)), ((double)((*a) + (*b))))));
#line 138 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 139 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 139 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z11Cutrates_kmffffffRfS_S_S_(
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_km, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_km, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_km, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_km, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_km, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *a, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *b, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ninf, 
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ntau){
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 140 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 142 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 143 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z10Curates_kmffffffRfS_S_S_(v, gbar_km, tha_km, qa_km, Ra_km, Rb_km, a, b, ninf, ntau);
#line 144 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 144 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z10Curates_kmffffffRfS_S_S_(
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_km, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_km, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_km, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_km, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_km, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *a, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *b, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ninf, 
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ntau){
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 145 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 147 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 148 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 149 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*a) = ((Ra_km * qa_km) * (_Z9Cuefun_kmf(((float)(fdividef(((double)(-(v - tha_km))), ((double)qa_km)))))));
#line 150 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 151 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*b) = ((Rb_km * qa_km) * (_Z9Cuefun_kmf(((float)(fdividef(((double)(v - tha_km)), ((double)qa_km)))))));
#line 153 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ntau) = ((float)(fdivide((0.31158821952278315), ((double)((*a) + (*b))))));
#line 154 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ninf) = ((float)(fdividef(((double)(*a)), ((double)((*a) + (*b))))));
#line 155 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 155 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z11Cutrates_kvffffffRfS_S_S_(
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kv, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_kv, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_kv, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kv, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kv, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *a, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *b, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ninf, 
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ntau){
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 156 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 158 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 159 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z10Curates_kvffffffRfS_S_S_(v, gbar_kv, tha_kv, qa_kv, Ra_kv, Rb_kv, a, b, ninf, ntau);
#line 160 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 160 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z10Curates_kvffffffRfS_S_S_(
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kv, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_kv, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_kv, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kv, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kv, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *a, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *b, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ninf, 
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ntau){
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 161 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 163 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 164 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 165 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*a) = ((Ra_kv * qa_kv) * (_Z9Cuefun_kvf(((float)(fdividef(((double)(-(v - tha_kv))), ((double)qa_kv)))))));
#line 166 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 167 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*b) = ((Rb_kv * qa_kv) * (_Z9Cuefun_kvf(((float)(fdividef(((double)(v - tha_kv)), ((double)qa_kv)))))));
#line 169 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ntau) = ((float)(fdivide((0.31158821952278315), ((double)((*a) + (*b))))));
#line 170 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ninf) = ((float)(fdividef(((double)(*a)), ((double)((*a) + (*b))))));
#line 171 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 171 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z11Cutrates_nafffffffffffffRfS_S_S_(
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi1_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi2_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qi_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thinf_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qinf_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rg_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rd_na, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *hinf, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *htau, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *minf, 
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *mtau){
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 172 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 174 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 175 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 176 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 177 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z10Curates_nafffffffffffffRfS_S_S_(v, gbar_na, tha_na, qa_na, Ra_na, Rb_na, thi1_na, thi2_na, qi_na, thinf_na, qinf_na, Rg_na, Rd_na, hinf, htau, minf, mtau);
#line 178 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 178 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z10Curates_nafffffffffffffRfS_S_S_(
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float vm, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi1_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi2_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qi_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thinf_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qinf_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rg_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rd_na, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *hinf, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *htau, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *minf, 
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *mtau){
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 179 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 180 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242176_8_non_const_a;
#line 180 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242176_11_non_const_b;
#line 181 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242176_8_non_const_a = (_Z10Cutrap0_naffff(vm, tha_na, Ra_na, qa_na));
#line 182 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242176_11_non_const_b = (_Z10Cutrap0_naffff((-vm), (-tha_na), Rb_na, qa_na));
#line 184 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*mtau) = ((float)(fdivide((0.31158821952278315), ((double)(__cuda_local_var_242176_8_non_const_a + __cuda_local_var_242176_11_non_const_b)))));
#line 185 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*minf) = ((float)(fdividef(((double)__cuda_local_var_242176_8_non_const_a), ((double)(__cuda_local_var_242176_8_non_const_a + __cuda_local_var_242176_11_non_const_b)))));
#line 186 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
;
#line 187 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242176_8_non_const_a = (_Z10Cutrap0_naffff(vm, thi1_na, Rd_na, qi_na));
#line 188 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242176_11_non_const_b = (_Z10Cutrap0_naffff((-vm), (-thi2_na), Rg_na, qi_na));
#line 189 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*htau) = ((float)(fdivide((0.31158821952278315), ((double)(__cuda_local_var_242176_8_non_const_a + __cuda_local_var_242176_11_non_const_b)))));
#line 190 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*hinf) = ((1.0F) / ((1.0F) + (_Z3expf(((float)(fdividef(((double)(vm - thinf_na)), ((double)qinf_na))))))));
#line 191 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 191 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z14CuInitModel_cafRfS_fffS_(
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *m, 
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *h, 
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_ca, 
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cao_ca, 
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cai, 
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ica){
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 196 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 197 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242193_7_non_const_hinf;
#line 197 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242193_12_non_const_htau;
#line 197 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242193_17_non_const_minf;
#line 197 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242193_22_non_const_mtau;
#line 199 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_cafffRfS_S_S_(((float)(((double)v) + (0.0))), gbar_ca, cao_ca, (&__cuda_local_var_242193_7_non_const_hinf), (&__cuda_local_var_242193_12_non_const_htau), (&__cuda_local_var_242193_17_non_const_minf), (&__cuda_local_var_242193_22_non_const_mtau));
#line 200 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*m) = __cuda_local_var_242193_17_non_const_minf;
#line 201 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*h) = __cuda_local_var_242193_7_non_const_hinf;
#line 202 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 202 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z15CuInitModel_cadfRffS_(
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ca, 
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float ica, 
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *cai){
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 205 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 206 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ca) = (9.999999747e-005F);
#line 207 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*cai) = (*ca);
#line 208 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 208 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z15CuInitModel_kcafRffffff(
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kca, 
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float caix_kca, 
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kca, 
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kca, 
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cai){
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 211 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 212 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242208_7_non_const_a;
#line 212 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242208_9_non_const_b;
#line 212 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242208_11_non_const_ninf;
#line 212 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242208_16_non_const_ntau;
#line 213 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Curates_kcafffffRfS_S_S_(cai, gbar_kca, caix_kca, Ra_kca, Rb_kca, (&__cuda_local_var_242208_7_non_const_a), (&__cuda_local_var_242208_9_non_const_b), (&__cuda_local_var_242208_11_non_const_ninf), (&__cuda_local_var_242208_16_non_const_ntau));
#line 214 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*n) = __cuda_local_var_242208_11_non_const_ninf;
#line 215 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 215 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z14CuInitModel_kmfRffffff(
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_km, 
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_km, 
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_km, 
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_km, 
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_km){
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 218 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 219 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242215_7_non_const_a;
#line 219 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242215_9_non_const_b;
#line 219 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242215_11_non_const_ninf;
#line 219 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242215_16_non_const_ntau;
#line 221 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_kmffffffRfS_S_S_(v, gbar_km, tha_km, qa_km, Ra_km, Rb_km, (&__cuda_local_var_242215_7_non_const_a), (&__cuda_local_var_242215_9_non_const_b), (&__cuda_local_var_242215_11_non_const_ninf), (&__cuda_local_var_242215_16_non_const_ntau));
#line 222 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*n) = __cuda_local_var_242215_11_non_const_ninf;
#line 223 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 223 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z14CuInitModel_kvfRffffff(
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kv, 
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_kv, 
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_kv, 
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kv, 
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kv){
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 226 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 227 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242223_7_non_const_a;
#line 227 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242223_9_non_const_b;
#line 227 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242223_11_non_const_ninf;
#line 227 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242223_16_non_const_ntau;
#line 229 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_kvffffffRfS_S_S_(v, gbar_kv, tha_kv, qa_kv, Ra_kv, Rb_kv, (&__cuda_local_var_242223_7_non_const_a), (&__cuda_local_var_242223_9_non_const_b), (&__cuda_local_var_242223_11_non_const_ninf), (&__cuda_local_var_242223_16_non_const_ntau));
#line 230 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*n) = __cuda_local_var_242223_11_non_const_ninf;
#line 231 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 231 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z14CuInitModel_nafRfS_ffffffffffff(
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *m, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *h, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi1_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi2_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qi_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thinf_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qinf_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rg_na, 
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rd_na){
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 234 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 235 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242231_7_non_const_hinf;
#line 235 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242231_12_non_const_htau;
#line 235 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242231_17_non_const_minf;
#line 235 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242231_22_non_const_mtau;
#line 237 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_nafffffffffffffRfS_S_S_(((float)(((double)v) + (-5.0))), gbar_na, tha_na, qa_na, Ra_na, Rb_na, thi1_na, thi2_na, qi_na, thinf_na, qinf_na, Rg_na, Rd_na, (&__cuda_local_var_242231_7_non_const_hinf), (&__cuda_local_var_242231_12_non_const_htau), (&__cuda_local_var_242231_17_non_const_minf), (&__cuda_local_var_242231_22_non_const_mtau));
#line 238 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*m) = __cuda_local_var_242231_17_non_const_minf;
#line 239 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*h) = __cuda_local_var_242231_7_non_const_hinf;
#line 240 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 240 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z15CuDerivModel_caffRfS_fffS_(
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float dt, 
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *m, 
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *h, 
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_ca, 
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cao_ca, 
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cai, 
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ica){
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 243 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 245 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242241_7_non_const_hinf;
#line 245 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242241_12_non_const_htau;
#line 245 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242241_17_non_const_minf;
#line 245 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242241_22_non_const_mtau;
#line 246 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_cafffRfS_S_S_(((float)(((double)v) + (0.0))), gbar_ca, cao_ca, (&__cuda_local_var_242241_7_non_const_hinf), (&__cuda_local_var_242241_12_non_const_htau), (&__cuda_local_var_242241_17_non_const_minf), (&__cuda_local_var_242241_22_non_const_mtau));
#line 247 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*m) = ((float)(((double)(*m)) + (((1.0) - (exp((((double)dt) * ((double)(fdivide((-1.0), ((double)__cuda_local_var_242241_22_non_const_mtau)))))))) * (((double)(fdivide(((double)(-(fdividef(((double)__cuda_local_var_242241_17_non_const_minf), ((double)__cuda_local_var_242241_22_non_const_mtau))))), (fdivide((-1.0), ((double)__cuda_local_var_242241_22_non_const_mtau)))))) - ((double)(*m))))));
#line 248 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*h) = ((float)(((double)(*h)) + (((1.0) - (exp((((double)dt) * ((double)(fdivide((-1.0), ((double)__cuda_local_var_242241_12_non_const_htau)))))))) * (((double)(fdivide(((double)(-(fdividef(((double)__cuda_local_var_242241_7_non_const_hinf), ((double)__cuda_local_var_242241_12_non_const_htau))))), (fdivide((-1.0), ((double)__cuda_local_var_242241_12_non_const_htau)))))) - ((double)(*h))))));
#line 249 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 249 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z16CuDerivModel_cadffRffS_(
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float dt, 
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ca, 
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float ica, 
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *cai){
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 251 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 252 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242248_7_non_const_drive_channel;
#line 253 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242248_7_non_const_drive_channel = ((float)(fdivide(((-10000.0) * ((double)ica)), (19297.0625))));
#line 254 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
if (((double)__cuda_local_var_242248_7_non_const_drive_channel) <= (0.0))
#line 254 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 255 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242248_7_non_const_drive_channel = (0.0F);
#line 256 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}
#line 257 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ca) = ((float)(((double)(*ca)) + (((1.0) - (exp((((double)dt) * (-0.0050000000000000001))))) * (((double)(fdivide((-(((double)__cuda_local_var_242248_7_non_const_drive_channel) + (4.9999999999999998e-007))), (-0.0050000000000000001)))) - ((double)(*ca))))));
#line 258 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*cai) = (*ca);
#line 259 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 259 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z16CuDerivModel_kcaffRffffff(
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float dt, 
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kca, 
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float caix_kca, 
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kca, 
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kca, 
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cai){
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 261 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 263 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242259_7_non_const_a;
#line 263 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242259_9_non_const_b;
#line 263 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242259_11_non_const_ninf;
#line 263 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242259_16_non_const_ntau;
#line 262 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(-90.0F);
#line 264 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Curates_kcafffffRfS_S_S_(cai, gbar_kca, caix_kca, Ra_kca, Rb_kca, (&__cuda_local_var_242259_7_non_const_a), (&__cuda_local_var_242259_9_non_const_b), (&__cuda_local_var_242259_11_non_const_ninf), (&__cuda_local_var_242259_16_non_const_ntau));
#line 265 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*n) = ((float)(((double)(*n)) + (((1.0) - (exp((((double)dt) * ((double)(fdivide((-1.0), ((double)__cuda_local_var_242259_16_non_const_ntau)))))))) * (((double)(fdivide(((double)(-(fdividef(((double)__cuda_local_var_242259_11_non_const_ninf), ((double)__cuda_local_var_242259_16_non_const_ntau))))), (fdivide((-1.0), ((double)__cuda_local_var_242259_16_non_const_ntau)))))) - ((double)(*n))))));
#line 266 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 266 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z15CuDerivModel_kmffRffffff(
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float dt, 
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_km, 
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_km, 
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_km, 
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_km, 
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_km){
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 268 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 270 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242266_7_non_const_a;
#line 270 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242266_9_non_const_b;
#line 270 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242266_11_non_const_ninf;
#line 270 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242266_16_non_const_ntau;
#line 271 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_kmffffffRfS_S_S_(v, gbar_km, tha_km, qa_km, Ra_km, Rb_km, (&__cuda_local_var_242266_7_non_const_a), (&__cuda_local_var_242266_9_non_const_b), (&__cuda_local_var_242266_11_non_const_ninf), (&__cuda_local_var_242266_16_non_const_ntau));
#line 272 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*n) = ((float)(((double)(*n)) + (((1.0) - (exp((((double)dt) * ((double)(fdivide((-1.0), ((double)__cuda_local_var_242266_16_non_const_ntau)))))))) * (((double)(fdivide(((double)(-(fdividef(((double)__cuda_local_var_242266_11_non_const_ninf), ((double)__cuda_local_var_242266_16_non_const_ntau))))), (fdivide((-1.0), ((double)__cuda_local_var_242266_16_non_const_ntau)))))) - ((double)(*n))))));
#line 273 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 273 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z15CuDerivModel_kvffRffffff(
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float dt, 
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kv, 
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_kv, 
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_kv, 
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kv, 
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kv){
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 275 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 277 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242273_7_non_const_a;
#line 277 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242273_9_non_const_b;
#line 277 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242273_11_non_const_ninf;
#line 277 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242273_16_non_const_ntau;
#line 278 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_kvffffffRfS_S_S_(v, gbar_kv, tha_kv, qa_kv, Ra_kv, Rb_kv, (&__cuda_local_var_242273_7_non_const_a), (&__cuda_local_var_242273_9_non_const_b), (&__cuda_local_var_242273_11_non_const_ninf), (&__cuda_local_var_242273_16_non_const_ntau));
#line 279 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*n) = ((float)(((double)(*n)) + (((1.0) - (exp((((double)dt) * ((double)(fdivide((-1.0), ((double)__cuda_local_var_242273_16_non_const_ntau)))))))) * (((double)(fdivide(((double)(-(fdividef(((double)__cuda_local_var_242273_11_non_const_ninf), ((double)__cuda_local_var_242273_16_non_const_ntau))))), (fdivide((-1.0), ((double)__cuda_local_var_242273_16_non_const_ntau)))))) - ((double)(*n))))));
#line 280 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 280 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z15CuDerivModel_naffRfS_ffffffffffff(
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float dt, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *m, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *h, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi1_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi2_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qi_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thinf_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qinf_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rg_na, 
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rd_na){
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 282 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 284 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242280_7_non_const_hinf;
#line 284 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242280_12_non_const_htau;
#line 284 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242280_17_non_const_minf;
#line 284 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242280_22_non_const_mtau;
#line 285 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
_Z11Cutrates_nafffffffffffffRfS_S_S_(((float)(((double)v) + (-5.0))), gbar_na, tha_na, qa_na, Ra_na, Rb_na, thi1_na, thi2_na, qi_na, thinf_na, qinf_na, Rg_na, Rd_na, (&__cuda_local_var_242280_7_non_const_hinf), (&__cuda_local_var_242280_12_non_const_htau), (&__cuda_local_var_242280_17_non_const_minf), (&__cuda_local_var_242280_22_non_const_mtau));
#line 286 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*m) = ((float)(((double)(*m)) + (((1.0) - (exp((((double)dt) * ((double)(fdivide((-1.0), ((double)__cuda_local_var_242280_22_non_const_mtau)))))))) * (((double)(fdivide(((double)(-(fdividef(((double)__cuda_local_var_242280_17_non_const_minf), ((double)__cuda_local_var_242280_22_non_const_mtau))))), (fdivide((-1.0), ((double)__cuda_local_var_242280_22_non_const_mtau)))))) - ((double)(*m))))));
#line 287 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*h) = ((float)(((double)(*h)) + (((1.0) - (exp((((double)dt) * ((double)(fdivide((-1.0), ((double)__cuda_local_var_242280_12_non_const_htau)))))))) * (((double)(fdivide(((double)(-(fdividef(((double)__cuda_local_var_242280_7_non_const_hinf), ((double)__cuda_local_var_242280_12_non_const_htau))))), (fdivide((-1.0), ((double)__cuda_local_var_242280_12_non_const_htau)))))) - ((double)(*h))))));
#line 288 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 288 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z20CuBreakpointModel_caRdRffS0_S0_fffS0_(
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
double *sumCurrents, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *sumConductivity, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *m, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *h, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_ca, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cao_ca, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cai, 
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ica){
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 294 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 295 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242291_7_non_const_gca;
#line 297 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242291_7_non_const_gca = ((float)(((((3.20936395327) * ((double)gbar_ca)) * ((double)(*m))) * ((double)(*m))) * ((double)(*h))));
#line 298 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*ica) = ((float)(((0.0001) * ((double)__cuda_local_var_242291_7_non_const_gca)) * ((double)(v - (140.0F)))));
#line 299 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumCurrents) += ((double)(*ica));
#line 300 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumConductivity) += __cuda_local_var_242291_7_non_const_gca;
#line 301 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 301 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z21CuBreakpointModel_cadRdRffS0_fS0_(
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
double *sumCurrents, 
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *sumConductivity, 
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *ca, 
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float ica, 
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *cai){
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 305 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 308 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 308 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z21CuBreakpointModel_kcaRdRffS0_fffff(
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
double *sumCurrents, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *sumConductivity, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kca, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float caix_kca, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kca, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kca, 
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float cai){
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 312 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 313 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242309_18_non_const_gk;
#line 314 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242310_7_non_const_ik;
#line 315 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242309_18_non_const_gk = ((float)(((3.20936395327) * ((double)gbar_kca)) * ((double)(*n))));
#line 316 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242310_7_non_const_ik = ((float)(((0.0001) * ((double)__cuda_local_var_242309_18_non_const_gk)) * ((double)(v - (-90.0F)))));
#line 317 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumCurrents) += ((double)__cuda_local_var_242310_7_non_const_ik);
#line 318 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumConductivity) += __cuda_local_var_242309_18_non_const_gk;
#line 319 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 319 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z20CuBreakpointModel_kmRdRffS0_fffff(
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
double *sumCurrents, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *sumConductivity, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_km, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_km, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_km, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_km, 
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_km){
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 323 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 324 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242320_13_non_const_gk;
#line 325 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242321_7_non_const_ik;
#line 326 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242320_13_non_const_gk = ((float)(((3.20936395327) * ((double)gbar_km)) * ((double)(*n))));
#line 327 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242321_7_non_const_ik = ((float)(((0.0001) * ((double)__cuda_local_var_242320_13_non_const_gk)) * ((double)(v - (-90.0F)))));
#line 328 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumCurrents) += ((double)__cuda_local_var_242321_7_non_const_ik);
#line 329 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumConductivity) += __cuda_local_var_242320_13_non_const_gk;
#line 330 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 330 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z20CuBreakpointModel_kvRdRffS0_fffff(
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
double *sumCurrents, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *sumConductivity, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *n, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_kv, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_kv, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_kv, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_kv, 
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_kv){
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 334 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 335 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242331_13_non_const_gk;
#line 336 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242332_7_non_const_ik;
#line 337 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242331_13_non_const_gk = ((float)(((3.20936395327) * ((double)gbar_kv)) * ((double)(*n))));
#line 338 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242332_7_non_const_ik = ((float)(((0.0001) * ((double)__cuda_local_var_242331_13_non_const_gk)) * ((double)(v - (-90.0F)))));
#line 339 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumCurrents) += ((double)__cuda_local_var_242332_7_non_const_ik);
#line 340 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumConductivity) += __cuda_local_var_242331_13_non_const_gk;
#line 341 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 341 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 void _Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff(
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
double *sumCurrents, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *sumConductivity, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float v, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *m, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float *h, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float gbar_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float tha_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qa_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Ra_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rb_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi1_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thi2_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qi_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float thinf_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float qinf_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rg_na, 
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
float Rd_na){
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 345 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
{
#line 346 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242342_13_non_const_gna;
#line 347 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
 float __cuda_local_var_242343_7_non_const_ina;
#line 348 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242342_13_non_const_gna = ((float)((((((3.20936395327) * ((double)gbar_na)) * ((double)(*m))) * ((double)(*m))) * ((double)(*m))) * ((double)(*h))));
#line 349 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
__cuda_local_var_242343_7_non_const_ina = ((float)(((0.0001) * ((double)__cuda_local_var_242342_13_non_const_gna)) * ((double)(v - (60.0F)))));
#line 350 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumCurrents) += ((double)__cuda_local_var_242343_7_non_const_ina);
#line 351 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
(*sumConductivity) += __cuda_local_var_242342_13_non_const_gna;
#line 352 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
} 
#line 352 "c:\\pyneurogpu_win\\neurogpu6\\AllModels.cu"
}}
#line 179 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 void _Z8BeforeLU4HMatPdS0_t(
#line 179 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
HMat InMat, 
#line 179 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
double *uHP, 
#line 179 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
double *bHP, 
#line 179 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
unsigned short Depth){
#line 179 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 180 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 181 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242610_17_non_const_PIdx;
#line 182 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242611_17_non_const_i;
#line 182 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242611_19_non_const_j;
#line 182 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242611_21_non_const_CurJ;
#line 182 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242611_26_non_const_CurB;
#line 182 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242611_31_non_const_t;
#line 182 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242611_34_non_const_CurLevel;
#line 182 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242611_43_non_const_LRelIndex;
#line 183 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242612_17_non_const_JumctionI;
#line 181 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242610_17_non_const_PIdx = ((unsigned short)(threadIdx.x));
#line 184 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242611_43_non_const_LRelIndex = ((cLRelStarts)[__cuda_local_var_242611_34_non_const_CurLevel]);
#line 185 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242611_43_non_const_LRelIndex = ((unsigned short)(((int)__cuda_local_var_242611_43_non_const_LRelIndex) + ((int)((cLRelEnds)[__cuda_local_var_242611_34_non_const_CurLevel])))); {
#line 186 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (__cuda_local_var_242611_34_non_const_CurLevel = ((unsigned short)0U); (((int)__cuda_local_var_242611_34_non_const_CurLevel) <= ((int)Depth)); __cuda_local_var_242611_34_non_const_CurLevel++)
#line 186 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 186 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 188 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (__cuda_local_var_242611_43_non_const_LRelIndex = ((cLRelStarts)[__cuda_local_var_242611_34_non_const_CurLevel]); (((int)__cuda_local_var_242611_43_non_const_LRelIndex) <= ((int)((cLRelEnds)[__cuda_local_var_242611_34_non_const_CurLevel]))); __cuda_local_var_242611_43_non_const_LRelIndex++)
#line 188 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 190 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242612_17_non_const_JumctionI = ((unsigned short)(((int)((cCompByLevel32)[((((int)__cuda_local_var_242611_43_non_const_LRelIndex) * 32) + ((int)__cuda_local_var_242610_17_non_const_PIdx))])) - 1)); {
#line 191 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (__cuda_local_var_242611_17_non_const_i = ((unsigned short)(((int)((cSegStartI)[__cuda_local_var_242612_17_non_const_JumctionI])) - 1)); (((int)__cuda_local_var_242611_17_non_const_i) < ((int)((cSegEndI)[__cuda_local_var_242612_17_non_const_JumctionI]))); __cuda_local_var_242611_17_non_const_i++)
#line 191 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 192 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242621_12_non_const_uHPm1;
#line 196 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242625_12_non_const_bHPm1;
#line 192 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242621_12_non_const_uHPm1 = (uHP[(((int)__cuda_local_var_242611_17_non_const_i) - 1)]);
#line 194 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(uHP[__cuda_local_var_242611_17_non_const_i]) = ((uHP[__cuda_local_var_242611_17_non_const_i]) - (((cF)[(((int)__cuda_local_var_242611_17_non_const_i) - 1)]) * ((double)(fdivide(((cE)[(((int)__cuda_local_var_242611_17_non_const_i) - 1)]), __cuda_local_var_242621_12_non_const_uHPm1)))));
#line 195 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242621_12_non_const_uHPm1 = (uHP[(((int)__cuda_local_var_242611_17_non_const_i) - 1)]);
#line 196 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242625_12_non_const_bHPm1 = (bHP[(((int)__cuda_local_var_242611_17_non_const_i) - 1)]);
#line 197 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(bHP[__cuda_local_var_242611_17_non_const_i]) = ((bHP[__cuda_local_var_242611_17_non_const_i]) - ((double)(fdivide((__cuda_local_var_242625_12_non_const_bHPm1 * ((cE)[(((int)__cuda_local_var_242611_17_non_const_i) - 1)])), __cuda_local_var_242621_12_non_const_uHPm1))));
#line 198 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 199 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 200 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242611_34_non_const_CurLevel) < ((int)Depth))
#line 200 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 200 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 201 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (__cuda_local_var_242611_43_non_const_LRelIndex = ((cFLRelStarts)[__cuda_local_var_242611_34_non_const_CurLevel]); (((int)__cuda_local_var_242611_43_non_const_LRelIndex) <= ((int)((cFLRelEnds)[__cuda_local_var_242611_34_non_const_CurLevel]))); __cuda_local_var_242611_43_non_const_LRelIndex++)
#line 201 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 204 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242633_20_non_const_St;
#line 205 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242634_20_non_const_En;
#line 202 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242611_26_non_const_CurB = ((unsigned short)(((int)((cCompByFLevel32)[((((int)__cuda_local_var_242611_43_non_const_LRelIndex) * 32) + ((int)__cuda_local_var_242610_17_non_const_PIdx))])) - 1));
#line 203 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242611_21_non_const_CurJ = ((unsigned short)(((int)((cFathers)[__cuda_local_var_242611_26_non_const_CurB])) - 1));
#line 204 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242633_20_non_const_St = ((cRelStarts)[__cuda_local_var_242611_26_non_const_CurB]);
#line 205 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242634_20_non_const_En = ((cRelEnds)[__cuda_local_var_242611_26_non_const_CurB]); {
#line 206 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (__cuda_local_var_242611_19_non_const_j = __cuda_local_var_242633_20_non_const_St; (((int)__cuda_local_var_242611_19_non_const_j) <= ((int)__cuda_local_var_242634_20_non_const_En)); __cuda_local_var_242611_19_non_const_j++)
#line 206 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 208 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242637_13_non_const_uHPm1;
#line 211 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242640_13_non_const_bHPm1;
#line 207 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242611_31_non_const_t = ((unsigned short)(((int)((cRelVec)[(((int)__cuda_local_var_242611_19_non_const_j) - 1)])) - 1));
#line 208 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242637_13_non_const_uHPm1 = (uHP[(((int)__cuda_local_var_242611_31_non_const_t) - 1)]);
#line 209 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(uHP[__cuda_local_var_242611_21_non_const_CurJ]) -= (((cF)[(((int)__cuda_local_var_242611_31_non_const_t) - 1)]) * ((double)(fdivide(((cE)[(((int)__cuda_local_var_242611_31_non_const_t) - 1)]), __cuda_local_var_242637_13_non_const_uHPm1))));
#line 210 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242637_13_non_const_uHPm1 = (uHP[(((int)__cuda_local_var_242611_31_non_const_t) - 1)]);
#line 211 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242640_13_non_const_bHPm1 = (bHP[(((int)__cuda_local_var_242611_31_non_const_t) - 1)]);
#line 212 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(bHP[__cuda_local_var_242611_21_non_const_CurJ]) -= ((double)(fdivide((__cuda_local_var_242640_13_non_const_bHPm1 * ((cE)[(((int)__cuda_local_var_242611_31_non_const_t) - 1)])), __cuda_local_var_242637_13_non_const_uHPm1)));
#line 213 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 214 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 215 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 216 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 217 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} 
#line 217 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}}
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 void _Z5BkSub4HMatPdS0_S0_S0_t(
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
HMat InMat, 
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
double *PX, 
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
double *PF, 
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
double *uHP, 
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
double *bHP, 
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
unsigned short LognDepth){
#line 220 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 221 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_16_non_const_PIdx_1;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_64_non_const_NextID_1;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_89_non_const_PIdx_2;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_137_non_const_NextID_2;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_162_non_const_PIdx_3;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_210_non_const_NextID_3;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_235_non_const_PIdx_4;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_283_non_const_NextID_4;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_308_non_const_PIdx_5;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_356_non_const_NextID_5;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_381_non_const_PIdx_6;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_429_non_const_NextID_6;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_454_non_const_PIdx_7;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_502_non_const_NextID_7;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_527_non_const_PIdx_8;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_575_non_const_NextID_8;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_600_non_const_PIdx_9;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_648_non_const_NextID_9;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_673_non_const_PIdx_10;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_723_non_const_NextID_10;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_749_non_const_PIdx_11;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_799_non_const_NextID_11;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_825_non_const_PIdx_12;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242653_875_non_const_NextID_12;
#line 225 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242654_17_non_const_i;
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_16_non_const_PIdx_1 = ((unsigned short)((threadIdx.x) + 0U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_89_non_const_PIdx_2 = ((unsigned short)((threadIdx.x) + 32U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_162_non_const_PIdx_3 = ((unsigned short)((threadIdx.x) + 64U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_235_non_const_PIdx_4 = ((unsigned short)((threadIdx.x) + 96U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_308_non_const_PIdx_5 = ((unsigned short)((threadIdx.x) + 128U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_381_non_const_PIdx_6 = ((unsigned short)((threadIdx.x) + 160U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_454_non_const_PIdx_7 = ((unsigned short)((threadIdx.x) + 192U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_527_non_const_PIdx_8 = ((unsigned short)((threadIdx.x) + 224U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_600_non_const_PIdx_9 = ((unsigned short)((threadIdx.x) + 256U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_673_non_const_PIdx_10 = ((unsigned short)((threadIdx.x) + 288U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_749_non_const_PIdx_11 = ((unsigned short)((threadIdx.x) + 320U));
#line 224 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_825_non_const_PIdx_12 = ((unsigned short)((threadIdx.x) + 352U));
#line 227 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
PX = bHP;
#line 228 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
PF = uHP;
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_16_non_const_PIdx_1]) = ((double)(fdivide((PX[__cuda_local_var_242653_16_non_const_PIdx_1]), (PF[__cuda_local_var_242653_16_non_const_PIdx_1]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_16_non_const_PIdx_1]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_16_non_const_PIdx_1])), (PF[__cuda_local_var_242653_16_non_const_PIdx_1]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_89_non_const_PIdx_2]) = ((double)(fdivide((PX[__cuda_local_var_242653_89_non_const_PIdx_2]), (PF[__cuda_local_var_242653_89_non_const_PIdx_2]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_89_non_const_PIdx_2]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_89_non_const_PIdx_2])), (PF[__cuda_local_var_242653_89_non_const_PIdx_2]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_162_non_const_PIdx_3]) = ((double)(fdivide((PX[__cuda_local_var_242653_162_non_const_PIdx_3]), (PF[__cuda_local_var_242653_162_non_const_PIdx_3]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_162_non_const_PIdx_3]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_162_non_const_PIdx_3])), (PF[__cuda_local_var_242653_162_non_const_PIdx_3]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_235_non_const_PIdx_4]) = ((double)(fdivide((PX[__cuda_local_var_242653_235_non_const_PIdx_4]), (PF[__cuda_local_var_242653_235_non_const_PIdx_4]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_235_non_const_PIdx_4]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_235_non_const_PIdx_4])), (PF[__cuda_local_var_242653_235_non_const_PIdx_4]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_308_non_const_PIdx_5]) = ((double)(fdivide((PX[__cuda_local_var_242653_308_non_const_PIdx_5]), (PF[__cuda_local_var_242653_308_non_const_PIdx_5]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_308_non_const_PIdx_5]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_308_non_const_PIdx_5])), (PF[__cuda_local_var_242653_308_non_const_PIdx_5]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_381_non_const_PIdx_6]) = ((double)(fdivide((PX[__cuda_local_var_242653_381_non_const_PIdx_6]), (PF[__cuda_local_var_242653_381_non_const_PIdx_6]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_381_non_const_PIdx_6]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_381_non_const_PIdx_6])), (PF[__cuda_local_var_242653_381_non_const_PIdx_6]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_454_non_const_PIdx_7]) = ((double)(fdivide((PX[__cuda_local_var_242653_454_non_const_PIdx_7]), (PF[__cuda_local_var_242653_454_non_const_PIdx_7]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_454_non_const_PIdx_7]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_454_non_const_PIdx_7])), (PF[__cuda_local_var_242653_454_non_const_PIdx_7]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_527_non_const_PIdx_8]) = ((double)(fdivide((PX[__cuda_local_var_242653_527_non_const_PIdx_8]), (PF[__cuda_local_var_242653_527_non_const_PIdx_8]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_527_non_const_PIdx_8]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_527_non_const_PIdx_8])), (PF[__cuda_local_var_242653_527_non_const_PIdx_8]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_600_non_const_PIdx_9]) = ((double)(fdivide((PX[__cuda_local_var_242653_600_non_const_PIdx_9]), (PF[__cuda_local_var_242653_600_non_const_PIdx_9]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_600_non_const_PIdx_9]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_600_non_const_PIdx_9])), (PF[__cuda_local_var_242653_600_non_const_PIdx_9]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_673_non_const_PIdx_10]) = ((double)(fdivide((PX[__cuda_local_var_242653_673_non_const_PIdx_10]), (PF[__cuda_local_var_242653_673_non_const_PIdx_10]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_673_non_const_PIdx_10]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_673_non_const_PIdx_10])), (PF[__cuda_local_var_242653_673_non_const_PIdx_10]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_749_non_const_PIdx_11]) = ((double)(fdivide((PX[__cuda_local_var_242653_749_non_const_PIdx_11]), (PF[__cuda_local_var_242653_749_non_const_PIdx_11]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_749_non_const_PIdx_11]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_749_non_const_PIdx_11])), (PF[__cuda_local_var_242653_749_non_const_PIdx_11]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_825_non_const_PIdx_12]) = ((double)(fdivide((PX[__cuda_local_var_242653_825_non_const_PIdx_12]), (PF[__cuda_local_var_242653_825_non_const_PIdx_12]))));
#line 231 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_825_non_const_PIdx_12]) = ((double)(fdivide((-((cF)[__cuda_local_var_242653_825_non_const_PIdx_12])), (PF[__cuda_local_var_242653_825_non_const_PIdx_12]))));
#line 233 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[(InMat.N)]) = (0.0);
#line 234 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[(InMat.N)]) = (1.0); {
#line 235 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (__cuda_local_var_242654_17_non_const_i = ((unsigned short)0U); (((int)__cuda_local_var_242654_17_non_const_i) < ((int)LognDepth)); __cuda_local_var_242654_17_non_const_i++)
#line 235 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_46_non_const_OldPXj_1;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_75_non_const_OldPXNextID_1;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_199_non_const_OldPXj_2;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_228_non_const_OldPXNextID_2;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_352_non_const_OldPXj_3;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_381_non_const_OldPXNextID_3;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_505_non_const_OldPXj_4;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_534_non_const_OldPXNextID_4;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_658_non_const_OldPXj_5;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_687_non_const_OldPXNextID_5;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_811_non_const_OldPXj_6;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_840_non_const_OldPXNextID_6;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_964_non_const_OldPXj_7;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_993_non_const_OldPXNextID_7;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1117_non_const_OldPXj_8;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1146_non_const_OldPXNextID_8;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1270_non_const_OldPXj_9;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1299_non_const_OldPXNextID_9;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1425_non_const_OldPXj_10;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1456_non_const_OldPXNextID_10;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1588_non_const_OldPXj_11;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1619_non_const_OldPXNextID_11;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1751_non_const_OldPXj_12;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242667_1782_non_const_OldPXNextID_12;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_7_non_const_OldPFj_1;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_36_non_const_OldPFNextID_1;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_109_non_const_OldPFj_2;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_138_non_const_OldPFNextID_2;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_211_non_const_OldPFj_3;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_240_non_const_OldPFNextID_3;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_313_non_const_OldPFj_4;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_342_non_const_OldPFNextID_4;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_415_non_const_OldPFj_5;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_444_non_const_OldPFNextID_5;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_517_non_const_OldPFj_6;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_546_non_const_OldPFNextID_6;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_619_non_const_OldPFj_7;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_648_non_const_OldPFNextID_7;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_721_non_const_OldPFj_8;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_750_non_const_OldPFNextID_8;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_823_non_const_OldPFj_9;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_852_non_const_OldPFNextID_9;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_925_non_const_OldPFj_10;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_956_non_const_OldPFNextID_10;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_1034_non_const_OldPFj_11;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_1065_non_const_OldPFNextID_11;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_1143_non_const_OldPFj_12;
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242673_1174_non_const_OldPFNextID_12;
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_64_non_const_NextID_1 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_16_non_const_PIdx_1))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_46_non_const_OldPXj_1 = ((float)(PX[__cuda_local_var_242653_16_non_const_PIdx_1]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_75_non_const_OldPXNextID_1 = ((float)(PX[__cuda_local_var_242653_64_non_const_NextID_1]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_16_non_const_PIdx_1]) = (((double)__cuda_local_var_242667_46_non_const_OldPXj_1) + (((double)__cuda_local_var_242667_75_non_const_OldPXNextID_1) * (PF[__cuda_local_var_242653_16_non_const_PIdx_1])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_137_non_const_NextID_2 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_89_non_const_PIdx_2))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_199_non_const_OldPXj_2 = ((float)(PX[__cuda_local_var_242653_89_non_const_PIdx_2]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_228_non_const_OldPXNextID_2 = ((float)(PX[__cuda_local_var_242653_137_non_const_NextID_2]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_89_non_const_PIdx_2]) = (((double)__cuda_local_var_242667_199_non_const_OldPXj_2) + (((double)__cuda_local_var_242667_228_non_const_OldPXNextID_2) * (PF[__cuda_local_var_242653_89_non_const_PIdx_2])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_210_non_const_NextID_3 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_162_non_const_PIdx_3))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_352_non_const_OldPXj_3 = ((float)(PX[__cuda_local_var_242653_162_non_const_PIdx_3]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_381_non_const_OldPXNextID_3 = ((float)(PX[__cuda_local_var_242653_210_non_const_NextID_3]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_162_non_const_PIdx_3]) = (((double)__cuda_local_var_242667_352_non_const_OldPXj_3) + (((double)__cuda_local_var_242667_381_non_const_OldPXNextID_3) * (PF[__cuda_local_var_242653_162_non_const_PIdx_3])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_283_non_const_NextID_4 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_235_non_const_PIdx_4))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_505_non_const_OldPXj_4 = ((float)(PX[__cuda_local_var_242653_235_non_const_PIdx_4]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_534_non_const_OldPXNextID_4 = ((float)(PX[__cuda_local_var_242653_283_non_const_NextID_4]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_235_non_const_PIdx_4]) = (((double)__cuda_local_var_242667_505_non_const_OldPXj_4) + (((double)__cuda_local_var_242667_534_non_const_OldPXNextID_4) * (PF[__cuda_local_var_242653_235_non_const_PIdx_4])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_356_non_const_NextID_5 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_308_non_const_PIdx_5))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_658_non_const_OldPXj_5 = ((float)(PX[__cuda_local_var_242653_308_non_const_PIdx_5]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_687_non_const_OldPXNextID_5 = ((float)(PX[__cuda_local_var_242653_356_non_const_NextID_5]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_308_non_const_PIdx_5]) = (((double)__cuda_local_var_242667_658_non_const_OldPXj_5) + (((double)__cuda_local_var_242667_687_non_const_OldPXNextID_5) * (PF[__cuda_local_var_242653_308_non_const_PIdx_5])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_429_non_const_NextID_6 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_381_non_const_PIdx_6))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_811_non_const_OldPXj_6 = ((float)(PX[__cuda_local_var_242653_381_non_const_PIdx_6]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_840_non_const_OldPXNextID_6 = ((float)(PX[__cuda_local_var_242653_429_non_const_NextID_6]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_381_non_const_PIdx_6]) = (((double)__cuda_local_var_242667_811_non_const_OldPXj_6) + (((double)__cuda_local_var_242667_840_non_const_OldPXNextID_6) * (PF[__cuda_local_var_242653_381_non_const_PIdx_6])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_502_non_const_NextID_7 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_454_non_const_PIdx_7))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_964_non_const_OldPXj_7 = ((float)(PX[__cuda_local_var_242653_454_non_const_PIdx_7]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_993_non_const_OldPXNextID_7 = ((float)(PX[__cuda_local_var_242653_502_non_const_NextID_7]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_454_non_const_PIdx_7]) = (((double)__cuda_local_var_242667_964_non_const_OldPXj_7) + (((double)__cuda_local_var_242667_993_non_const_OldPXNextID_7) * (PF[__cuda_local_var_242653_454_non_const_PIdx_7])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_575_non_const_NextID_8 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_527_non_const_PIdx_8))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1117_non_const_OldPXj_8 = ((float)(PX[__cuda_local_var_242653_527_non_const_PIdx_8]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1146_non_const_OldPXNextID_8 = ((float)(PX[__cuda_local_var_242653_575_non_const_NextID_8]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_527_non_const_PIdx_8]) = (((double)__cuda_local_var_242667_1117_non_const_OldPXj_8) + (((double)__cuda_local_var_242667_1146_non_const_OldPXNextID_8) * (PF[__cuda_local_var_242653_527_non_const_PIdx_8])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_648_non_const_NextID_9 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_600_non_const_PIdx_9))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1270_non_const_OldPXj_9 = ((float)(PX[__cuda_local_var_242653_600_non_const_PIdx_9]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1299_non_const_OldPXNextID_9 = ((float)(PX[__cuda_local_var_242653_648_non_const_NextID_9]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_600_non_const_PIdx_9]) = (((double)__cuda_local_var_242667_1270_non_const_OldPXj_9) + (((double)__cuda_local_var_242667_1299_non_const_OldPXNextID_9) * (PF[__cuda_local_var_242653_600_non_const_PIdx_9])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_723_non_const_NextID_10 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_673_non_const_PIdx_10))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1425_non_const_OldPXj_10 = ((float)(PX[__cuda_local_var_242653_673_non_const_PIdx_10]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1456_non_const_OldPXNextID_10 = ((float)(PX[__cuda_local_var_242653_723_non_const_NextID_10]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_673_non_const_PIdx_10]) = (((double)__cuda_local_var_242667_1425_non_const_OldPXj_10) + (((double)__cuda_local_var_242667_1456_non_const_OldPXNextID_10) * (PF[__cuda_local_var_242653_673_non_const_PIdx_10])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_799_non_const_NextID_11 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_749_non_const_PIdx_11))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1588_non_const_OldPXj_11 = ((float)(PX[__cuda_local_var_242653_749_non_const_PIdx_11]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1619_non_const_OldPXNextID_11 = ((float)(PX[__cuda_local_var_242653_799_non_const_NextID_11]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_749_non_const_PIdx_11]) = (((double)__cuda_local_var_242667_1588_non_const_OldPXj_11) + (((double)__cuda_local_var_242667_1619_non_const_OldPXNextID_11) * (PF[__cuda_local_var_242653_749_non_const_PIdx_11])));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242653_875_non_const_NextID_12 = ((unsigned short)(((int)((cFIdxs)[((((int)__cuda_local_var_242654_17_non_const_i) * ((int)(InMat.N))) + ((int)__cuda_local_var_242653_825_non_const_PIdx_12))])) - 1));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1751_non_const_OldPXj_12 = ((float)(PX[__cuda_local_var_242653_825_non_const_PIdx_12]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242667_1782_non_const_OldPXNextID_12 = ((float)(PX[__cuda_local_var_242653_875_non_const_NextID_12]));
#line 238 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PX[__cuda_local_var_242653_825_non_const_PIdx_12]) = (((double)__cuda_local_var_242667_1751_non_const_OldPXj_12) + (((double)__cuda_local_var_242667_1782_non_const_OldPXNextID_12) * (PF[__cuda_local_var_242653_825_non_const_PIdx_12])));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_7_non_const_OldPFj_1 = ((float)(PF[__cuda_local_var_242653_16_non_const_PIdx_1]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_36_non_const_OldPFNextID_1 = ((float)(PF[__cuda_local_var_242653_64_non_const_NextID_1]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_16_non_const_PIdx_1]) = ((double)(__cuda_local_var_242673_7_non_const_OldPFj_1 * __cuda_local_var_242673_36_non_const_OldPFNextID_1));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_109_non_const_OldPFj_2 = ((float)(PF[__cuda_local_var_242653_89_non_const_PIdx_2]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_138_non_const_OldPFNextID_2 = ((float)(PF[__cuda_local_var_242653_137_non_const_NextID_2]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_89_non_const_PIdx_2]) = ((double)(__cuda_local_var_242673_109_non_const_OldPFj_2 * __cuda_local_var_242673_138_non_const_OldPFNextID_2));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_211_non_const_OldPFj_3 = ((float)(PF[__cuda_local_var_242653_162_non_const_PIdx_3]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_240_non_const_OldPFNextID_3 = ((float)(PF[__cuda_local_var_242653_210_non_const_NextID_3]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_162_non_const_PIdx_3]) = ((double)(__cuda_local_var_242673_211_non_const_OldPFj_3 * __cuda_local_var_242673_240_non_const_OldPFNextID_3));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_313_non_const_OldPFj_4 = ((float)(PF[__cuda_local_var_242653_235_non_const_PIdx_4]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_342_non_const_OldPFNextID_4 = ((float)(PF[__cuda_local_var_242653_283_non_const_NextID_4]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_235_non_const_PIdx_4]) = ((double)(__cuda_local_var_242673_313_non_const_OldPFj_4 * __cuda_local_var_242673_342_non_const_OldPFNextID_4));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_415_non_const_OldPFj_5 = ((float)(PF[__cuda_local_var_242653_308_non_const_PIdx_5]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_444_non_const_OldPFNextID_5 = ((float)(PF[__cuda_local_var_242653_356_non_const_NextID_5]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_308_non_const_PIdx_5]) = ((double)(__cuda_local_var_242673_415_non_const_OldPFj_5 * __cuda_local_var_242673_444_non_const_OldPFNextID_5));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_517_non_const_OldPFj_6 = ((float)(PF[__cuda_local_var_242653_381_non_const_PIdx_6]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_546_non_const_OldPFNextID_6 = ((float)(PF[__cuda_local_var_242653_429_non_const_NextID_6]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_381_non_const_PIdx_6]) = ((double)(__cuda_local_var_242673_517_non_const_OldPFj_6 * __cuda_local_var_242673_546_non_const_OldPFNextID_6));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_619_non_const_OldPFj_7 = ((float)(PF[__cuda_local_var_242653_454_non_const_PIdx_7]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_648_non_const_OldPFNextID_7 = ((float)(PF[__cuda_local_var_242653_502_non_const_NextID_7]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_454_non_const_PIdx_7]) = ((double)(__cuda_local_var_242673_619_non_const_OldPFj_7 * __cuda_local_var_242673_648_non_const_OldPFNextID_7));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_721_non_const_OldPFj_8 = ((float)(PF[__cuda_local_var_242653_527_non_const_PIdx_8]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_750_non_const_OldPFNextID_8 = ((float)(PF[__cuda_local_var_242653_575_non_const_NextID_8]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_527_non_const_PIdx_8]) = ((double)(__cuda_local_var_242673_721_non_const_OldPFj_8 * __cuda_local_var_242673_750_non_const_OldPFNextID_8));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_823_non_const_OldPFj_9 = ((float)(PF[__cuda_local_var_242653_600_non_const_PIdx_9]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_852_non_const_OldPFNextID_9 = ((float)(PF[__cuda_local_var_242653_648_non_const_NextID_9]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_600_non_const_PIdx_9]) = ((double)(__cuda_local_var_242673_823_non_const_OldPFj_9 * __cuda_local_var_242673_852_non_const_OldPFNextID_9));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_925_non_const_OldPFj_10 = ((float)(PF[__cuda_local_var_242653_673_non_const_PIdx_10]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_956_non_const_OldPFNextID_10 = ((float)(PF[__cuda_local_var_242653_723_non_const_NextID_10]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_673_non_const_PIdx_10]) = ((double)(__cuda_local_var_242673_925_non_const_OldPFj_10 * __cuda_local_var_242673_956_non_const_OldPFNextID_10));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_1034_non_const_OldPFj_11 = ((float)(PF[__cuda_local_var_242653_749_non_const_PIdx_11]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_1065_non_const_OldPFNextID_11 = ((float)(PF[__cuda_local_var_242653_799_non_const_NextID_11]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_749_non_const_PIdx_11]) = ((double)(__cuda_local_var_242673_1034_non_const_OldPFj_11 * __cuda_local_var_242673_1065_non_const_OldPFNextID_11));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_1143_non_const_OldPFj_12 = ((float)(PF[__cuda_local_var_242653_825_non_const_PIdx_12]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242673_1174_non_const_OldPFNextID_12 = ((float)(PF[__cuda_local_var_242653_875_non_const_NextID_12]));
#line 244 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(PF[__cuda_local_var_242653_825_non_const_PIdx_12]) = ((double)(__cuda_local_var_242673_1143_non_const_OldPFj_12 * __cuda_local_var_242673_1174_non_const_OldPFNextID_12));
#line 245 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 246 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} 
#line 246 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}}
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 void _Z13runSimulation4HMatPfS0_4Stim3SimS0_S0_S0_t(
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
HMat InMat, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
float *ParamsM, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
float *V, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
Stim stim, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
Sim sim, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
float *VHotGlobal, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
float *SMemVHot, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
float *amps, 
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
unsigned short offset){
#line 285 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 286 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 289 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double *__cuda_local_var_242718_10_non_const_uHP;
#line 289 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double *__cuda_local_var_242718_15_non_const_bHP;
#line 290 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242719_17_non_const_StimID;
#line 297 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242726_18_non_const_PerStimulus;
#line 318 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242747_17_non_const_NeuronID;
#line 319 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 int __cuda_local_var_242748_6_non_const_Nt;
#line 320 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242749_8_non_const_t;
#line 321 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double *__cuda_local_var_242750_10_non_const_PX;
#line 321 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double *__cuda_local_var_242750_14_non_const_PF;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_16_non_const_PIdx_1;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_63_non_const_PIdx_2;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_110_non_const_PIdx_3;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_157_non_const_PIdx_4;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_204_non_const_PIdx_5;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_251_non_const_PIdx_6;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_298_non_const_PIdx_7;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_345_non_const_PIdx_8;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_392_non_const_PIdx_9;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_439_non_const_PIdx_10;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_488_non_const_PIdx_11;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242754_537_non_const_PIdx_12;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_28_non_const_Vmid_1;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_43_non_const_ModelStates_1[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_69_non_const_v_1;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_92_non_const_dv_1;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_124_non_const_Vmid_2;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_139_non_const_ModelStates_2[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_165_non_const_v_2;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_188_non_const_dv_2;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_220_non_const_Vmid_3;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_235_non_const_ModelStates_3[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_261_non_const_v_3;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_284_non_const_dv_3;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_316_non_const_Vmid_4;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_331_non_const_ModelStates_4[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_357_non_const_v_4;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_380_non_const_dv_4;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_412_non_const_Vmid_5;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_427_non_const_ModelStates_5[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_453_non_const_v_5;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_476_non_const_dv_5;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_508_non_const_Vmid_6;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_523_non_const_ModelStates_6[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_549_non_const_v_6;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_572_non_const_dv_6;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_604_non_const_Vmid_7;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_619_non_const_ModelStates_7[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_645_non_const_v_7;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_668_non_const_dv_7;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_700_non_const_Vmid_8;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_715_non_const_ModelStates_8[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_741_non_const_v_8;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_764_non_const_dv_8;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_796_non_const_Vmid_9;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_811_non_const_ModelStates_9[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_837_non_const_v_9;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_860_non_const_dv_9;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_894_non_const_Vmid_10;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_910_non_const_ModelStates_10[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_937_non_const_v_10;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_962_non_const_dv_10;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_997_non_const_Vmid_11;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_1013_non_const_ModelStates_11[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_1040_non_const_v_11;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_1065_non_const_dv_11;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_1100_non_const_Vmid_12;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_1116_non_const_ModelStates_12[10];
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_1143_non_const_v_12;
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242757_1168_non_const_dv_12;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_8_non_const_sumCurrents_1;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_25_non_const_sumCurrentsDv_1;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_51_non_const_sumConductivity_1;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_72_non_const_sumConductivityDv_1;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_102_non_const_sumCurrents_2;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_119_non_const_sumCurrentsDv_2;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_145_non_const_sumConductivity_2;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_166_non_const_sumConductivityDv_2;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_196_non_const_sumCurrents_3;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_213_non_const_sumCurrentsDv_3;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_239_non_const_sumConductivity_3;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_260_non_const_sumConductivityDv_3;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_290_non_const_sumCurrents_4;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_307_non_const_sumCurrentsDv_4;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_333_non_const_sumConductivity_4;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_354_non_const_sumConductivityDv_4;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_384_non_const_sumCurrents_5;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_401_non_const_sumCurrentsDv_5;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_427_non_const_sumConductivity_5;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_448_non_const_sumConductivityDv_5;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_478_non_const_sumCurrents_6;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_495_non_const_sumCurrentsDv_6;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_521_non_const_sumConductivity_6;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_542_non_const_sumConductivityDv_6;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_572_non_const_sumCurrents_7;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_589_non_const_sumCurrentsDv_7;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_615_non_const_sumConductivity_7;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_636_non_const_sumConductivityDv_7;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_666_non_const_sumCurrents_8;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_683_non_const_sumCurrentsDv_8;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_709_non_const_sumConductivity_8;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_730_non_const_sumConductivityDv_8;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_760_non_const_sumCurrents_9;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_777_non_const_sumCurrentsDv_9;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_803_non_const_sumConductivity_9;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_824_non_const_sumConductivityDv_9;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_854_non_const_sumCurrents_10;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_872_non_const_sumCurrentsDv_10;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_899_non_const_sumConductivity_10;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_921_non_const_sumConductivityDv_10;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_952_non_const_sumCurrents_11;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_970_non_const_sumCurrentsDv_11;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_997_non_const_sumConductivity_11;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_1019_non_const_sumConductivityDv_11;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_1050_non_const_sumCurrents_12;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242760_1068_non_const_sumCurrentsDv_12;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_1095_non_const_sumConductivity_12;
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242760_1117_non_const_sumConductivityDv_12;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_16_non_const_SonNo_1;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_59_non_const_SonNo_2;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_102_non_const_SonNo_3;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_145_non_const_SonNo_4;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_188_non_const_SonNo_5;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_231_non_const_SonNo_6;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_274_non_const_SonNo_7;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_317_non_const_SonNo_8;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_360_non_const_SonNo_9;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_403_non_const_SonNo_10;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_448_non_const_SonNo_11;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242764_493_non_const_SonNo_12;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_16_non_const_parentIndex_1;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_47_non_const_Eidx_1;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_98_non_const_parentIndex_2;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_129_non_const_Eidx_2;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_180_non_const_parentIndex_3;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_211_non_const_Eidx_3;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_262_non_const_parentIndex_4;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_293_non_const_Eidx_4;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_344_non_const_parentIndex_5;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_375_non_const_Eidx_5;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_426_non_const_parentIndex_6;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_457_non_const_Eidx_6;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_508_non_const_parentIndex_7;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_539_non_const_Eidx_7;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_590_non_const_parentIndex_8;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_621_non_const_Eidx_8;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_672_non_const_parentIndex_9;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_703_non_const_Eidx_9;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_754_non_const_parentIndex_10;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_786_non_const_Eidx_10;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_840_non_const_parentIndex_11;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_872_non_const_Eidx_11;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_926_non_const_parentIndex_12;
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242771_958_non_const_Eidx_12;
#line 347 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242776_17_non_const_perThreadParamMSize;
#line 355 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242784_17_non_const_stimLoc;
#line 356 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242785_11_non_const_stimArea;
#line 357 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 unsigned short __cuda_local_var_242786_20_non_const_dtCounter;
#line 358 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242787_12_non_const_dt;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_8_non_const_rhs_1;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_15_non_const_D_1;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_26_non_const_gModel_1;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_36_non_const_StimCurrent_1;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_57_non_const_rhs_2;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_64_non_const_D_2;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_75_non_const_gModel_2;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_85_non_const_StimCurrent_2;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_106_non_const_rhs_3;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_113_non_const_D_3;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_124_non_const_gModel_3;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_134_non_const_StimCurrent_3;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_155_non_const_rhs_4;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_162_non_const_D_4;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_173_non_const_gModel_4;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_183_non_const_StimCurrent_4;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_204_non_const_rhs_5;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_211_non_const_D_5;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_222_non_const_gModel_5;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_232_non_const_StimCurrent_5;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_253_non_const_rhs_6;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_260_non_const_D_6;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_271_non_const_gModel_6;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_281_non_const_StimCurrent_6;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_302_non_const_rhs_7;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_309_non_const_D_7;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_320_non_const_gModel_7;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_330_non_const_StimCurrent_7;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_351_non_const_rhs_8;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_358_non_const_D_8;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_369_non_const_gModel_8;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_379_non_const_StimCurrent_8;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_400_non_const_rhs_9;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_407_non_const_D_9;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_418_non_const_gModel_9;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_428_non_const_StimCurrent_9;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_449_non_const_rhs_10;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_457_non_const_D_10;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_469_non_const_gModel_10;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_480_non_const_StimCurrent_10;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_502_non_const_rhs_11;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_510_non_const_D_11;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_522_non_const_gModel_11;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_533_non_const_StimCurrent_11;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_555_non_const_rhs_12;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 double __cuda_local_var_242795_563_non_const_D_12;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_575_non_const_gModel_12;
#line 366 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 float __cuda_local_var_242795_586_non_const_StimCurrent_12;
#line 290 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242719_17_non_const_StimID = ((unsigned short)(threadIdx.y));
#line 297 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242726_18_non_const_PerStimulus = ((unsigned short)((((unsigned long long)((((int)(InMat.N)) + 2) * 2)) * 8ULL) + (((unsigned long long)(32 + (((int)(sim.NRecSites)) * 32))) * 4ULL)));
#line 298 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242726_18_non_const_PerStimulus = ((unsigned short)(__float2uint_rz(((double)((float)((ceilf(((float)(fdivide(((double)__cuda_local_var_242726_18_non_const_PerStimulus), (8.0)))))) * (8.0F)))))));
#line 302 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
offset += (((unsigned)__cuda_local_var_242726_18_non_const_PerStimulus) * (threadIdx.y));
#line 303 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242718_10_non_const_uHP = ((double *)((smem) + offset));
#line 304 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
offset += (((unsigned long long)(((int)(InMat.N)) + 2)) * 8ULL);
#line 306 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242718_15_non_const_bHP = ((double *)((smem) + offset));
#line 307 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
offset += (((unsigned long long)(((int)(InMat.N)) + 2)) * 8ULL);
#line 309 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
offset = ((unsigned short)(__float2uint_rz(((double)((float)((ceilf(((float)(fdivide(((double)offset), (8.0)))))) * (8.0F)))))));
#line 313 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
SMemVHot = ((float *)((smem) + offset));
#line 314 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
offset += (((unsigned long long)(32 * ((int)(sim.NRecSites)))) * 4ULL);
#line 315 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(SMemVHot[32]) = (0.0F);
#line 316 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
amps = ((float *)((smem) + offset));
#line 317 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
offset += 128ULL;
#line 318 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242747_17_non_const_NeuronID = ((unsigned short)(blockIdx.x));
#line 319 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242748_6_non_const_Nt = ((int)(__float2int_rz(((double)((float)(stim.Nt))))));
#line 320 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242749_8_non_const_t = (0.0F);
#line 322 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242750_10_non_const_PX = __cuda_local_var_242718_15_non_const_bHP;
#line 323 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242750_14_non_const_PF = __cuda_local_var_242718_10_non_const_uHP;
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_16_non_const_PIdx_1 = ((unsigned short)((threadIdx.x) + 0U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_63_non_const_PIdx_2 = ((unsigned short)((threadIdx.x) + 32U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_110_non_const_PIdx_3 = ((unsigned short)((threadIdx.x) + 64U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_157_non_const_PIdx_4 = ((unsigned short)((threadIdx.x) + 96U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_204_non_const_PIdx_5 = ((unsigned short)((threadIdx.x) + 128U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_251_non_const_PIdx_6 = ((unsigned short)((threadIdx.x) + 160U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_298_non_const_PIdx_7 = ((unsigned short)((threadIdx.x) + 192U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_345_non_const_PIdx_8 = ((unsigned short)((threadIdx.x) + 224U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_392_non_const_PIdx_9 = ((unsigned short)((threadIdx.x) + 256U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_439_non_const_PIdx_10 = ((unsigned short)((threadIdx.x) + 288U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_488_non_const_PIdx_11 = ((unsigned short)((threadIdx.x) + 320U));
#line 325 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242754_537_non_const_PIdx_12 = ((unsigned short)((threadIdx.x) + 352U));
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_69_non_const_v_1 = (V[__cuda_local_var_242754_16_non_const_PIdx_1]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_165_non_const_v_2 = (V[__cuda_local_var_242754_63_non_const_PIdx_2]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_261_non_const_v_3 = (V[__cuda_local_var_242754_110_non_const_PIdx_3]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_357_non_const_v_4 = (V[__cuda_local_var_242754_157_non_const_PIdx_4]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_453_non_const_v_5 = (V[__cuda_local_var_242754_204_non_const_PIdx_5]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_549_non_const_v_6 = (V[__cuda_local_var_242754_251_non_const_PIdx_6]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_645_non_const_v_7 = (V[__cuda_local_var_242754_298_non_const_PIdx_7]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_741_non_const_v_8 = (V[__cuda_local_var_242754_345_non_const_PIdx_8]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_837_non_const_v_9 = (V[__cuda_local_var_242754_392_non_const_PIdx_9]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_937_non_const_v_10 = (V[__cuda_local_var_242754_439_non_const_PIdx_10]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1040_non_const_v_11 = (V[__cuda_local_var_242754_488_non_const_PIdx_11]);
#line 328 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1143_non_const_v_12 = (V[__cuda_local_var_242754_537_non_const_PIdx_12]);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_8_non_const_sumCurrents_1 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_25_non_const_sumCurrentsDv_1 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_51_non_const_sumConductivity_1 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_72_non_const_sumConductivityDv_1 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_102_non_const_sumCurrents_2 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_119_non_const_sumCurrentsDv_2 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_145_non_const_sumConductivity_2 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_166_non_const_sumConductivityDv_2 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_196_non_const_sumCurrents_3 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_213_non_const_sumCurrentsDv_3 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_239_non_const_sumConductivity_3 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_260_non_const_sumConductivityDv_3 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_290_non_const_sumCurrents_4 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_307_non_const_sumCurrentsDv_4 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_333_non_const_sumConductivity_4 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_354_non_const_sumConductivityDv_4 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_384_non_const_sumCurrents_5 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_401_non_const_sumCurrentsDv_5 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_427_non_const_sumConductivity_5 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_448_non_const_sumConductivityDv_5 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_478_non_const_sumCurrents_6 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_495_non_const_sumCurrentsDv_6 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_521_non_const_sumConductivity_6 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_542_non_const_sumConductivityDv_6 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_572_non_const_sumCurrents_7 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_589_non_const_sumCurrentsDv_7 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_615_non_const_sumConductivity_7 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_636_non_const_sumConductivityDv_7 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_666_non_const_sumCurrents_8 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_683_non_const_sumCurrentsDv_8 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_709_non_const_sumConductivity_8 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_730_non_const_sumConductivityDv_8 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_760_non_const_sumCurrents_9 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_777_non_const_sumCurrentsDv_9 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_803_non_const_sumConductivity_9 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_824_non_const_sumConductivityDv_9 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_854_non_const_sumCurrents_10 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_872_non_const_sumCurrentsDv_10 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_899_non_const_sumConductivity_10 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_921_non_const_sumConductivityDv_10 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_952_non_const_sumCurrents_11 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_970_non_const_sumCurrentsDv_11 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_997_non_const_sumConductivity_11 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1019_non_const_sumConductivityDv_11 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1050_non_const_sumCurrents_12 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12 = (0.0);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1095_non_const_sumConductivity_12 = (0.0F);
#line 331 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1117_non_const_sumConductivityDv_12 = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_43_non_const_ModelStates_1)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_139_non_const_ModelStates_2)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_235_non_const_ModelStates_3)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_331_non_const_ModelStates_4)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_427_non_const_ModelStates_5)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_523_non_const_ModelStates_6)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_619_non_const_ModelStates_7)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_715_non_const_ModelStates_8)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_811_non_const_ModelStates_9)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_910_non_const_ModelStates_10)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1013_non_const_ModelStates_11)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[0]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[1]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[2]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[3]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[4]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[5]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[6]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[7]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[8]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
((__cuda_local_var_242757_1116_non_const_ModelStates_12)[9]) = (0.0F);
#line 333 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_16_non_const_SonNo_1 = ((cSonNoVec)[__cuda_local_var_242754_16_non_const_PIdx_1]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_59_non_const_SonNo_2 = ((cSonNoVec)[__cuda_local_var_242754_63_non_const_PIdx_2]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_102_non_const_SonNo_3 = ((cSonNoVec)[__cuda_local_var_242754_110_non_const_PIdx_3]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_145_non_const_SonNo_4 = ((cSonNoVec)[__cuda_local_var_242754_157_non_const_PIdx_4]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_188_non_const_SonNo_5 = ((cSonNoVec)[__cuda_local_var_242754_204_non_const_PIdx_5]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_231_non_const_SonNo_6 = ((cSonNoVec)[__cuda_local_var_242754_251_non_const_PIdx_6]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_274_non_const_SonNo_7 = ((cSonNoVec)[__cuda_local_var_242754_298_non_const_PIdx_7]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_317_non_const_SonNo_8 = ((cSonNoVec)[__cuda_local_var_242754_345_non_const_PIdx_8]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_360_non_const_SonNo_9 = ((cSonNoVec)[__cuda_local_var_242754_392_non_const_PIdx_9]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_403_non_const_SonNo_10 = ((cSonNoVec)[__cuda_local_var_242754_439_non_const_PIdx_10]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_448_non_const_SonNo_11 = ((cSonNoVec)[__cuda_local_var_242754_488_non_const_PIdx_11]);
#line 335 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242764_493_non_const_SonNo_12 = ((cSonNoVec)[__cuda_local_var_242754_537_non_const_PIdx_12]);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_16_non_const_PIdx_1]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_92_non_const_dv_1 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_63_non_const_PIdx_2]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_188_non_const_dv_2 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_110_non_const_PIdx_3]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_284_non_const_dv_3 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_157_non_const_PIdx_4]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_380_non_const_dv_4 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_204_non_const_PIdx_5]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_476_non_const_dv_5 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_251_non_const_PIdx_6]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_572_non_const_dv_6 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_298_non_const_PIdx_7]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_668_non_const_dv_7 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_345_non_const_PIdx_8]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_764_non_const_dv_8 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_392_non_const_PIdx_9]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_860_non_const_dv_9 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_439_non_const_PIdx_10]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_962_non_const_dv_10 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_488_non_const_PIdx_11]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1065_non_const_dv_11 = (0.0F);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[__cuda_local_var_242754_537_non_const_PIdx_12]) = (0.0);
#line 338 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1168_non_const_dv_12 = (0.0F);
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_47_non_const_Eidx_1 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_129_non_const_Eidx_2 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_211_non_const_Eidx_3 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_293_non_const_Eidx_4 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_375_non_const_Eidx_5 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_457_non_const_Eidx_6 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_539_non_const_Eidx_7 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_621_non_const_Eidx_8 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_703_non_const_Eidx_9 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_786_non_const_Eidx_10 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_872_non_const_Eidx_11 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11)) - 1));
#line 342 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_958_non_const_Eidx_12 = ((unsigned short)((((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12)) - 1));
#line 344 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242771_47_non_const_Eidx_1) > (((int)(InMat.N)) - 1))
#line 344 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 345 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_47_non_const_Eidx_1 = ((unsigned short)(((int)(InMat.N)) - 1));
#line 346 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 347 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242776_17_non_const_perThreadParamMSize = ((unsigned short)(((int)(InMat.NComps)) * 28));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 0), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 2), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[9]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 6), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 0), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 2), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[9]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 6), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 0), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[8]), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 2), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[9]), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 6), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 0), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[8]), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 2), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[9]), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 6), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 0), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[8]), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 2), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[9]), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 6), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 0), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[8]), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 2), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[9]), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 6), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 0), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[8]), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 2), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[9]), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 6), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 0), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[8]), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 2), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[9]), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 6), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 0), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[8]), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 2), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[9]), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 6), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 0), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[8]), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 2), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[9]), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 6), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 0), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[8]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 2), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[9]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 6), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (0 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_cafRfS_fffS_(__cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 0), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[8]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 9));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (1 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_cadfRffS_(__cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 2), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[9]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 8));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (2 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuInitModel_kcafRffffff(__cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[8]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (3 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kmfRffffff(__cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (4 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_kvfRffffff(__cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (5 * ((int)(InMat.N))))])
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z14CuInitModel_nafRfS_ffffffffffff(__cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 6), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 350 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_16_non_const_parentIndex_1 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_98_non_const_parentIndex_2 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_180_non_const_parentIndex_3 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_262_non_const_parentIndex_4 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_344_non_const_parentIndex_5 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_426_non_const_parentIndex_6 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_508_non_const_parentIndex_7 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_590_non_const_parentIndex_8 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_672_non_const_parentIndex_9 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_754_non_const_parentIndex_10 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_840_non_const_parentIndex_11 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11))]))));
#line 354 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_926_non_const_parentIndex_12 = ((unsigned short)(((int)(InMat.N)) - ((int)((cKs)[(((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12))]))));
#line 355 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242784_17_non_const_stimLoc = (stim.loc);
#line 356 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242785_11_non_const_stimArea = (stim.area);
#line 357 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242786_20_non_const_dtCounter = ((unsigned short)0U);
#line 358 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242787_12_non_const_dt = (sim.dt);
#line 358 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 368 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 int i;
#line 368 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
i = 0; {
#line 368 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (; (i < __cuda_local_var_242748_6_non_const_Nt); i++)
#line 368 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 369 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (i == ((int)((stim.dtInds)[__cuda_local_var_242786_20_non_const_dtCounter])))
#line 369 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 370 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242787_12_non_const_dt = ((stim.durs)[__cuda_local_var_242786_20_non_const_dtCounter]);
#line 371 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242786_20_non_const_dtCounter) != (((int)(stim.numofdts)) - 1))
#line 371 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 372 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242786_20_non_const_dtCounter++;
#line 373 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 374 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 375 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242749_8_non_const_t += ((0.5) * ((double)__cuda_local_var_242787_12_non_const_dt));
#line 377 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((i % 32) == 0)
#line 377 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 378 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (i > 0)
#line 378 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 378 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 379 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 int recInd;
#line 379 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
recInd = 0; {
#line 379 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (; (recInd < ((int)(sim.NRecSites))); recInd++)
#line 379 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 380 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(VHotGlobal[(((((((unsigned)__cuda_local_var_242747_17_non_const_NeuronID) * (((unsigned)(((int)(sim.NRecSites)) * __cuda_local_var_242748_6_non_const_Nt)) * (blockDim.y))) + (((threadIdx.y) * ((unsigned)__cuda_local_var_242748_6_non_const_Nt)) * ((unsigned)(sim.NRecSites)))) + ((unsigned)(recInd * __cuda_local_var_242748_6_non_const_Nt))) + ((unsigned)(i - 32))) + ((unsigned)__cuda_local_var_242754_16_non_const_PIdx_1))]) = (SMemVHot[((32 * recInd) + ((int)__cuda_local_var_242754_16_non_const_PIdx_1))]);
#line 381 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 381 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 382 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 384 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(amps[__cuda_local_var_242754_16_non_const_PIdx_1]) = ((stim.amps)[((((threadIdx.y) * ((unsigned)__cuda_local_var_242748_6_non_const_Nt)) + ((unsigned)i)) + ((unsigned)__cuda_local_var_242754_16_non_const_PIdx_1))]);
#line 385 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 385 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 386 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
 int recInd;
#line 386 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
recInd = 0; {
#line 386 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
for (; (recInd < ((int)(sim.NRecSites))); recInd++)
#line 386 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 387 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((unsigned)(((int)((sim.RecSites)[recInd])) % 32)) == (threadIdx.x))
#line 387 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 389 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(SMemVHot[((recInd * 32) + (i % 32))]) = __cuda_local_var_242757_69_non_const_v_1;
#line 389 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 391 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
} }
#line 391 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_8_non_const_rhs_1 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_15_non_const_D_1 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_8_non_const_sumCurrents_1 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_51_non_const_sumConductivity_1 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_25_non_const_sumCurrentsDv_1 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_72_non_const_sumConductivityDv_1 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_36_non_const_StimCurrent_1 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_57_non_const_rhs_2 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_64_non_const_D_2 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_102_non_const_sumCurrents_2 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_145_non_const_sumConductivity_2 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_119_non_const_sumCurrentsDv_2 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_166_non_const_sumConductivityDv_2 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_85_non_const_StimCurrent_2 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_106_non_const_rhs_3 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_113_non_const_D_3 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_196_non_const_sumCurrents_3 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_239_non_const_sumConductivity_3 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_213_non_const_sumCurrentsDv_3 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_260_non_const_sumConductivityDv_3 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_134_non_const_StimCurrent_3 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_155_non_const_rhs_4 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_162_non_const_D_4 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_290_non_const_sumCurrents_4 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_333_non_const_sumConductivity_4 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_307_non_const_sumCurrentsDv_4 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_354_non_const_sumConductivityDv_4 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_183_non_const_StimCurrent_4 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_204_non_const_rhs_5 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_211_non_const_D_5 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_384_non_const_sumCurrents_5 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_427_non_const_sumConductivity_5 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_401_non_const_sumCurrentsDv_5 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_448_non_const_sumConductivityDv_5 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_232_non_const_StimCurrent_5 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_253_non_const_rhs_6 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_260_non_const_D_6 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_478_non_const_sumCurrents_6 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_521_non_const_sumConductivity_6 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_495_non_const_sumCurrentsDv_6 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_542_non_const_sumConductivityDv_6 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_281_non_const_StimCurrent_6 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_302_non_const_rhs_7 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_309_non_const_D_7 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_572_non_const_sumCurrents_7 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_615_non_const_sumConductivity_7 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_589_non_const_sumCurrentsDv_7 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_636_non_const_sumConductivityDv_7 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_330_non_const_StimCurrent_7 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_351_non_const_rhs_8 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_358_non_const_D_8 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_666_non_const_sumCurrents_8 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_709_non_const_sumConductivity_8 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_683_non_const_sumCurrentsDv_8 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_730_non_const_sumConductivityDv_8 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_379_non_const_StimCurrent_8 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_400_non_const_rhs_9 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_407_non_const_D_9 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_760_non_const_sumCurrents_9 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_803_non_const_sumConductivity_9 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_777_non_const_sumCurrentsDv_9 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_824_non_const_sumConductivityDv_9 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_428_non_const_StimCurrent_9 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_449_non_const_rhs_10 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_457_non_const_D_10 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_854_non_const_sumCurrents_10 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_899_non_const_sumConductivity_10 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_872_non_const_sumCurrentsDv_10 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_921_non_const_sumConductivityDv_10 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_480_non_const_StimCurrent_10 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_502_non_const_rhs_11 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_510_non_const_D_11 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_952_non_const_sumCurrents_11 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_997_non_const_sumConductivity_11 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_970_non_const_sumCurrentsDv_11 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1019_non_const_sumConductivityDv_11 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_533_non_const_StimCurrent_11 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_555_non_const_rhs_12 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_563_non_const_D_12 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1050_non_const_sumCurrents_12 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1095_non_const_sumConductivity_12 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12 = (0.0);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242760_1117_non_const_sumConductivityDv_12 = (0.0F);
#line 394 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_586_non_const_StimCurrent_12 = (0.0F);
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_16_non_const_PIdx_1) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_36_non_const_StimCurrent_1 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_63_non_const_PIdx_2) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_85_non_const_StimCurrent_2 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_110_non_const_PIdx_3) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_134_non_const_StimCurrent_3 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_157_non_const_PIdx_4) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_183_non_const_StimCurrent_4 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_204_non_const_PIdx_5) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_232_non_const_StimCurrent_5 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_251_non_const_PIdx_6) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_281_non_const_StimCurrent_6 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_298_non_const_PIdx_7) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_330_non_const_StimCurrent_7 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_345_non_const_PIdx_8) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_379_non_const_StimCurrent_8 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_392_non_const_PIdx_9) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_428_non_const_StimCurrent_9 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_439_non_const_PIdx_10) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_480_non_const_StimCurrent_10 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_488_non_const_PIdx_11) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_533_non_const_StimCurrent_11 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_537_non_const_PIdx_12) == ((int)__cuda_local_var_242784_17_non_const_stimLoc))
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_586_non_const_StimCurrent_12 = ((float)(fdividef(((double)((100.0F) * (amps[(i % 32)]))), ((double)__cuda_local_var_242785_11_non_const_stimArea))));
#line 404 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_25_non_const_sumCurrentsDv_1), (&__cuda_local_var_242760_72_non_const_sumConductivityDv_1), ((float)(((double)__cuda_local_var_242757_69_non_const_v_1) + (0.001))), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 0), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_25_non_const_sumCurrentsDv_1), (&__cuda_local_var_242760_72_non_const_sumConductivityDv_1), ((float)(((double)__cuda_local_var_242757_69_non_const_v_1) + (0.001))), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 2), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[9]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_25_non_const_sumCurrentsDv_1), (&__cuda_local_var_242760_72_non_const_sumConductivityDv_1), ((float)(((double)__cuda_local_var_242757_69_non_const_v_1) + (0.001))), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_25_non_const_sumCurrentsDv_1), (&__cuda_local_var_242760_72_non_const_sumConductivityDv_1), ((float)(((double)__cuda_local_var_242757_69_non_const_v_1) + (0.001))), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_25_non_const_sumCurrentsDv_1), (&__cuda_local_var_242760_72_non_const_sumConductivityDv_1), ((float)(((double)__cuda_local_var_242757_69_non_const_v_1) + (0.001))), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_25_non_const_sumCurrentsDv_1), (&__cuda_local_var_242760_72_non_const_sumConductivityDv_1), ((float)(((double)__cuda_local_var_242757_69_non_const_v_1) + (0.001))), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 6), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_119_non_const_sumCurrentsDv_2), (&__cuda_local_var_242760_166_non_const_sumConductivityDv_2), ((float)(((double)__cuda_local_var_242757_165_non_const_v_2) + (0.001))), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 0), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_119_non_const_sumCurrentsDv_2), (&__cuda_local_var_242760_166_non_const_sumConductivityDv_2), ((float)(((double)__cuda_local_var_242757_165_non_const_v_2) + (0.001))), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 2), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[9]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_119_non_const_sumCurrentsDv_2), (&__cuda_local_var_242760_166_non_const_sumConductivityDv_2), ((float)(((double)__cuda_local_var_242757_165_non_const_v_2) + (0.001))), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_119_non_const_sumCurrentsDv_2), (&__cuda_local_var_242760_166_non_const_sumConductivityDv_2), ((float)(((double)__cuda_local_var_242757_165_non_const_v_2) + (0.001))), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_119_non_const_sumCurrentsDv_2), (&__cuda_local_var_242760_166_non_const_sumConductivityDv_2), ((float)(((double)__cuda_local_var_242757_165_non_const_v_2) + (0.001))), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_119_non_const_sumCurrentsDv_2), (&__cuda_local_var_242760_166_non_const_sumConductivityDv_2), ((float)(((double)__cuda_local_var_242757_165_non_const_v_2) + (0.001))), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 6), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_213_non_const_sumCurrentsDv_3), (&__cuda_local_var_242760_260_non_const_sumConductivityDv_3), ((float)(((double)__cuda_local_var_242757_261_non_const_v_3) + (0.001))), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 0), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[8]), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_213_non_const_sumCurrentsDv_3), (&__cuda_local_var_242760_260_non_const_sumConductivityDv_3), ((float)(((double)__cuda_local_var_242757_261_non_const_v_3) + (0.001))), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 2), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[9]), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_213_non_const_sumCurrentsDv_3), (&__cuda_local_var_242760_260_non_const_sumConductivityDv_3), ((float)(((double)__cuda_local_var_242757_261_non_const_v_3) + (0.001))), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_213_non_const_sumCurrentsDv_3), (&__cuda_local_var_242760_260_non_const_sumConductivityDv_3), ((float)(((double)__cuda_local_var_242757_261_non_const_v_3) + (0.001))), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_213_non_const_sumCurrentsDv_3), (&__cuda_local_var_242760_260_non_const_sumConductivityDv_3), ((float)(((double)__cuda_local_var_242757_261_non_const_v_3) + (0.001))), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_213_non_const_sumCurrentsDv_3), (&__cuda_local_var_242760_260_non_const_sumConductivityDv_3), ((float)(((double)__cuda_local_var_242757_261_non_const_v_3) + (0.001))), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 6), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_307_non_const_sumCurrentsDv_4), (&__cuda_local_var_242760_354_non_const_sumConductivityDv_4), ((float)(((double)__cuda_local_var_242757_357_non_const_v_4) + (0.001))), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 0), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[8]), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_307_non_const_sumCurrentsDv_4), (&__cuda_local_var_242760_354_non_const_sumConductivityDv_4), ((float)(((double)__cuda_local_var_242757_357_non_const_v_4) + (0.001))), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 2), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[9]), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_307_non_const_sumCurrentsDv_4), (&__cuda_local_var_242760_354_non_const_sumConductivityDv_4), ((float)(((double)__cuda_local_var_242757_357_non_const_v_4) + (0.001))), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_307_non_const_sumCurrentsDv_4), (&__cuda_local_var_242760_354_non_const_sumConductivityDv_4), ((float)(((double)__cuda_local_var_242757_357_non_const_v_4) + (0.001))), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_307_non_const_sumCurrentsDv_4), (&__cuda_local_var_242760_354_non_const_sumConductivityDv_4), ((float)(((double)__cuda_local_var_242757_357_non_const_v_4) + (0.001))), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_307_non_const_sumCurrentsDv_4), (&__cuda_local_var_242760_354_non_const_sumConductivityDv_4), ((float)(((double)__cuda_local_var_242757_357_non_const_v_4) + (0.001))), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 6), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_401_non_const_sumCurrentsDv_5), (&__cuda_local_var_242760_448_non_const_sumConductivityDv_5), ((float)(((double)__cuda_local_var_242757_453_non_const_v_5) + (0.001))), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 0), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[8]), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_401_non_const_sumCurrentsDv_5), (&__cuda_local_var_242760_448_non_const_sumConductivityDv_5), ((float)(((double)__cuda_local_var_242757_453_non_const_v_5) + (0.001))), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 2), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[9]), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_401_non_const_sumCurrentsDv_5), (&__cuda_local_var_242760_448_non_const_sumConductivityDv_5), ((float)(((double)__cuda_local_var_242757_453_non_const_v_5) + (0.001))), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_401_non_const_sumCurrentsDv_5), (&__cuda_local_var_242760_448_non_const_sumConductivityDv_5), ((float)(((double)__cuda_local_var_242757_453_non_const_v_5) + (0.001))), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_401_non_const_sumCurrentsDv_5), (&__cuda_local_var_242760_448_non_const_sumConductivityDv_5), ((float)(((double)__cuda_local_var_242757_453_non_const_v_5) + (0.001))), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_401_non_const_sumCurrentsDv_5), (&__cuda_local_var_242760_448_non_const_sumConductivityDv_5), ((float)(((double)__cuda_local_var_242757_453_non_const_v_5) + (0.001))), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 6), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_495_non_const_sumCurrentsDv_6), (&__cuda_local_var_242760_542_non_const_sumConductivityDv_6), ((float)(((double)__cuda_local_var_242757_549_non_const_v_6) + (0.001))), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 0), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[8]), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_495_non_const_sumCurrentsDv_6), (&__cuda_local_var_242760_542_non_const_sumConductivityDv_6), ((float)(((double)__cuda_local_var_242757_549_non_const_v_6) + (0.001))), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 2), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[9]), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_495_non_const_sumCurrentsDv_6), (&__cuda_local_var_242760_542_non_const_sumConductivityDv_6), ((float)(((double)__cuda_local_var_242757_549_non_const_v_6) + (0.001))), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_495_non_const_sumCurrentsDv_6), (&__cuda_local_var_242760_542_non_const_sumConductivityDv_6), ((float)(((double)__cuda_local_var_242757_549_non_const_v_6) + (0.001))), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_495_non_const_sumCurrentsDv_6), (&__cuda_local_var_242760_542_non_const_sumConductivityDv_6), ((float)(((double)__cuda_local_var_242757_549_non_const_v_6) + (0.001))), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_495_non_const_sumCurrentsDv_6), (&__cuda_local_var_242760_542_non_const_sumConductivityDv_6), ((float)(((double)__cuda_local_var_242757_549_non_const_v_6) + (0.001))), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 6), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_589_non_const_sumCurrentsDv_7), (&__cuda_local_var_242760_636_non_const_sumConductivityDv_7), ((float)(((double)__cuda_local_var_242757_645_non_const_v_7) + (0.001))), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 0), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[8]), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_589_non_const_sumCurrentsDv_7), (&__cuda_local_var_242760_636_non_const_sumConductivityDv_7), ((float)(((double)__cuda_local_var_242757_645_non_const_v_7) + (0.001))), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 2), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[9]), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_589_non_const_sumCurrentsDv_7), (&__cuda_local_var_242760_636_non_const_sumConductivityDv_7), ((float)(((double)__cuda_local_var_242757_645_non_const_v_7) + (0.001))), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_589_non_const_sumCurrentsDv_7), (&__cuda_local_var_242760_636_non_const_sumConductivityDv_7), ((float)(((double)__cuda_local_var_242757_645_non_const_v_7) + (0.001))), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_589_non_const_sumCurrentsDv_7), (&__cuda_local_var_242760_636_non_const_sumConductivityDv_7), ((float)(((double)__cuda_local_var_242757_645_non_const_v_7) + (0.001))), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_589_non_const_sumCurrentsDv_7), (&__cuda_local_var_242760_636_non_const_sumConductivityDv_7), ((float)(((double)__cuda_local_var_242757_645_non_const_v_7) + (0.001))), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 6), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_683_non_const_sumCurrentsDv_8), (&__cuda_local_var_242760_730_non_const_sumConductivityDv_8), ((float)(((double)__cuda_local_var_242757_741_non_const_v_8) + (0.001))), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 0), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[8]), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_683_non_const_sumCurrentsDv_8), (&__cuda_local_var_242760_730_non_const_sumConductivityDv_8), ((float)(((double)__cuda_local_var_242757_741_non_const_v_8) + (0.001))), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 2), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[9]), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_683_non_const_sumCurrentsDv_8), (&__cuda_local_var_242760_730_non_const_sumConductivityDv_8), ((float)(((double)__cuda_local_var_242757_741_non_const_v_8) + (0.001))), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_683_non_const_sumCurrentsDv_8), (&__cuda_local_var_242760_730_non_const_sumConductivityDv_8), ((float)(((double)__cuda_local_var_242757_741_non_const_v_8) + (0.001))), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_683_non_const_sumCurrentsDv_8), (&__cuda_local_var_242760_730_non_const_sumConductivityDv_8), ((float)(((double)__cuda_local_var_242757_741_non_const_v_8) + (0.001))), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_683_non_const_sumCurrentsDv_8), (&__cuda_local_var_242760_730_non_const_sumConductivityDv_8), ((float)(((double)__cuda_local_var_242757_741_non_const_v_8) + (0.001))), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 6), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_777_non_const_sumCurrentsDv_9), (&__cuda_local_var_242760_824_non_const_sumConductivityDv_9), ((float)(((double)__cuda_local_var_242757_837_non_const_v_9) + (0.001))), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 0), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[8]), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_777_non_const_sumCurrentsDv_9), (&__cuda_local_var_242760_824_non_const_sumConductivityDv_9), ((float)(((double)__cuda_local_var_242757_837_non_const_v_9) + (0.001))), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 2), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[9]), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_777_non_const_sumCurrentsDv_9), (&__cuda_local_var_242760_824_non_const_sumConductivityDv_9), ((float)(((double)__cuda_local_var_242757_837_non_const_v_9) + (0.001))), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_777_non_const_sumCurrentsDv_9), (&__cuda_local_var_242760_824_non_const_sumConductivityDv_9), ((float)(((double)__cuda_local_var_242757_837_non_const_v_9) + (0.001))), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_777_non_const_sumCurrentsDv_9), (&__cuda_local_var_242760_824_non_const_sumConductivityDv_9), ((float)(((double)__cuda_local_var_242757_837_non_const_v_9) + (0.001))), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_777_non_const_sumCurrentsDv_9), (&__cuda_local_var_242760_824_non_const_sumConductivityDv_9), ((float)(((double)__cuda_local_var_242757_837_non_const_v_9) + (0.001))), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 6), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_872_non_const_sumCurrentsDv_10), (&__cuda_local_var_242760_921_non_const_sumConductivityDv_10), ((float)(((double)__cuda_local_var_242757_937_non_const_v_10) + (0.001))), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 0), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[8]), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_872_non_const_sumCurrentsDv_10), (&__cuda_local_var_242760_921_non_const_sumConductivityDv_10), ((float)(((double)__cuda_local_var_242757_937_non_const_v_10) + (0.001))), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 2), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[9]), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_872_non_const_sumCurrentsDv_10), (&__cuda_local_var_242760_921_non_const_sumConductivityDv_10), ((float)(((double)__cuda_local_var_242757_937_non_const_v_10) + (0.001))), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_872_non_const_sumCurrentsDv_10), (&__cuda_local_var_242760_921_non_const_sumConductivityDv_10), ((float)(((double)__cuda_local_var_242757_937_non_const_v_10) + (0.001))), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_872_non_const_sumCurrentsDv_10), (&__cuda_local_var_242760_921_non_const_sumConductivityDv_10), ((float)(((double)__cuda_local_var_242757_937_non_const_v_10) + (0.001))), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_872_non_const_sumCurrentsDv_10), (&__cuda_local_var_242760_921_non_const_sumConductivityDv_10), ((float)(((double)__cuda_local_var_242757_937_non_const_v_10) + (0.001))), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 6), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_970_non_const_sumCurrentsDv_11), (&__cuda_local_var_242760_1019_non_const_sumConductivityDv_11), ((float)(((double)__cuda_local_var_242757_1040_non_const_v_11) + (0.001))), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 0), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[8]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_970_non_const_sumCurrentsDv_11), (&__cuda_local_var_242760_1019_non_const_sumConductivityDv_11), ((float)(((double)__cuda_local_var_242757_1040_non_const_v_11) + (0.001))), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 2), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[9]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_970_non_const_sumCurrentsDv_11), (&__cuda_local_var_242760_1019_non_const_sumConductivityDv_11), ((float)(((double)__cuda_local_var_242757_1040_non_const_v_11) + (0.001))), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_970_non_const_sumCurrentsDv_11), (&__cuda_local_var_242760_1019_non_const_sumConductivityDv_11), ((float)(((double)__cuda_local_var_242757_1040_non_const_v_11) + (0.001))), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_970_non_const_sumCurrentsDv_11), (&__cuda_local_var_242760_1019_non_const_sumConductivityDv_11), ((float)(((double)__cuda_local_var_242757_1040_non_const_v_11) + (0.001))), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_970_non_const_sumCurrentsDv_11), (&__cuda_local_var_242760_1019_non_const_sumConductivityDv_11), ((float)(((double)__cuda_local_var_242757_1040_non_const_v_11) + (0.001))), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 6), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (0 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12), (&__cuda_local_var_242760_1117_non_const_sumConductivityDv_12), ((float)(((double)__cuda_local_var_242757_1143_non_const_v_12) + (0.001))), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 0), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[8]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 9));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (1 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12), (&__cuda_local_var_242760_1117_non_const_sumConductivityDv_12), ((float)(((double)__cuda_local_var_242757_1143_non_const_v_12) + (0.001))), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 2), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[9]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 8));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (2 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12), (&__cuda_local_var_242760_1117_non_const_sumConductivityDv_12), ((float)(((double)__cuda_local_var_242757_1143_non_const_v_12) + (0.001))), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[8]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (3 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12), (&__cuda_local_var_242760_1117_non_const_sumConductivityDv_12), ((float)(((double)__cuda_local_var_242757_1143_non_const_v_12) + (0.001))), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (4 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12), (&__cuda_local_var_242760_1117_non_const_sumConductivityDv_12), ((float)(((double)__cuda_local_var_242757_1143_non_const_v_12) + (0.001))), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (5 * ((int)(InMat.N))))])
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12), (&__cuda_local_var_242760_1117_non_const_sumConductivityDv_12), ((float)(((double)__cuda_local_var_242757_1143_non_const_v_12) + (0.001))), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 6), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 406 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_8_non_const_sumCurrents_1), (&__cuda_local_var_242760_51_non_const_sumConductivity_1), __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 0), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_8_non_const_sumCurrents_1), (&__cuda_local_var_242760_51_non_const_sumConductivity_1), __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 2), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[9]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_8_non_const_sumCurrents_1), (&__cuda_local_var_242760_51_non_const_sumConductivity_1), __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_8_non_const_sumCurrents_1), (&__cuda_local_var_242760_51_non_const_sumConductivity_1), __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_8_non_const_sumCurrents_1), (&__cuda_local_var_242760_51_non_const_sumConductivity_1), __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_8_non_const_sumCurrents_1), (&__cuda_local_var_242760_51_non_const_sumConductivity_1), __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 6), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_102_non_const_sumCurrents_2), (&__cuda_local_var_242760_145_non_const_sumConductivity_2), __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 0), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_102_non_const_sumCurrents_2), (&__cuda_local_var_242760_145_non_const_sumConductivity_2), __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 2), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[9]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_102_non_const_sumCurrents_2), (&__cuda_local_var_242760_145_non_const_sumConductivity_2), __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_102_non_const_sumCurrents_2), (&__cuda_local_var_242760_145_non_const_sumConductivity_2), __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_102_non_const_sumCurrents_2), (&__cuda_local_var_242760_145_non_const_sumConductivity_2), __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_102_non_const_sumCurrents_2), (&__cuda_local_var_242760_145_non_const_sumConductivity_2), __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 6), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_196_non_const_sumCurrents_3), (&__cuda_local_var_242760_239_non_const_sumConductivity_3), __cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 0), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[8]), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_196_non_const_sumCurrents_3), (&__cuda_local_var_242760_239_non_const_sumConductivity_3), __cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 2), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[9]), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_196_non_const_sumCurrents_3), (&__cuda_local_var_242760_239_non_const_sumConductivity_3), __cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), ((__cuda_local_var_242757_235_non_const_ModelStates_3)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_196_non_const_sumCurrents_3), (&__cuda_local_var_242760_239_non_const_sumConductivity_3), __cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_196_non_const_sumCurrents_3), (&__cuda_local_var_242760_239_non_const_sumConductivity_3), __cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_110_non_const_PIdx_3) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_196_non_const_sumCurrents_3), (&__cuda_local_var_242760_239_non_const_sumConductivity_3), __cuda_local_var_242757_261_non_const_v_3, ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 6), ((__cuda_local_var_242757_235_non_const_ModelStates_3) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_110_non_const_PIdx_3])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_290_non_const_sumCurrents_4), (&__cuda_local_var_242760_333_non_const_sumConductivity_4), __cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 0), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[8]), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_290_non_const_sumCurrents_4), (&__cuda_local_var_242760_333_non_const_sumConductivity_4), __cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 2), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[9]), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_290_non_const_sumCurrents_4), (&__cuda_local_var_242760_333_non_const_sumConductivity_4), __cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), ((__cuda_local_var_242757_331_non_const_ModelStates_4)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_290_non_const_sumCurrents_4), (&__cuda_local_var_242760_333_non_const_sumConductivity_4), __cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_290_non_const_sumCurrents_4), (&__cuda_local_var_242760_333_non_const_sumConductivity_4), __cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_157_non_const_PIdx_4) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_290_non_const_sumCurrents_4), (&__cuda_local_var_242760_333_non_const_sumConductivity_4), __cuda_local_var_242757_357_non_const_v_4, ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 6), ((__cuda_local_var_242757_331_non_const_ModelStates_4) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_157_non_const_PIdx_4])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_384_non_const_sumCurrents_5), (&__cuda_local_var_242760_427_non_const_sumConductivity_5), __cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 0), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[8]), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_384_non_const_sumCurrents_5), (&__cuda_local_var_242760_427_non_const_sumConductivity_5), __cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 2), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[9]), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_384_non_const_sumCurrents_5), (&__cuda_local_var_242760_427_non_const_sumConductivity_5), __cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), ((__cuda_local_var_242757_427_non_const_ModelStates_5)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_384_non_const_sumCurrents_5), (&__cuda_local_var_242760_427_non_const_sumConductivity_5), __cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_384_non_const_sumCurrents_5), (&__cuda_local_var_242760_427_non_const_sumConductivity_5), __cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_204_non_const_PIdx_5) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_384_non_const_sumCurrents_5), (&__cuda_local_var_242760_427_non_const_sumConductivity_5), __cuda_local_var_242757_453_non_const_v_5, ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 6), ((__cuda_local_var_242757_427_non_const_ModelStates_5) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_204_non_const_PIdx_5])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_478_non_const_sumCurrents_6), (&__cuda_local_var_242760_521_non_const_sumConductivity_6), __cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 0), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[8]), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_478_non_const_sumCurrents_6), (&__cuda_local_var_242760_521_non_const_sumConductivity_6), __cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 2), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[9]), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_478_non_const_sumCurrents_6), (&__cuda_local_var_242760_521_non_const_sumConductivity_6), __cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), ((__cuda_local_var_242757_523_non_const_ModelStates_6)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_478_non_const_sumCurrents_6), (&__cuda_local_var_242760_521_non_const_sumConductivity_6), __cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_478_non_const_sumCurrents_6), (&__cuda_local_var_242760_521_non_const_sumConductivity_6), __cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_251_non_const_PIdx_6) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_478_non_const_sumCurrents_6), (&__cuda_local_var_242760_521_non_const_sumConductivity_6), __cuda_local_var_242757_549_non_const_v_6, ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 6), ((__cuda_local_var_242757_523_non_const_ModelStates_6) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_251_non_const_PIdx_6])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_572_non_const_sumCurrents_7), (&__cuda_local_var_242760_615_non_const_sumConductivity_7), __cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 0), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[8]), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_572_non_const_sumCurrents_7), (&__cuda_local_var_242760_615_non_const_sumConductivity_7), __cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 2), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[9]), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_572_non_const_sumCurrents_7), (&__cuda_local_var_242760_615_non_const_sumConductivity_7), __cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), ((__cuda_local_var_242757_619_non_const_ModelStates_7)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_572_non_const_sumCurrents_7), (&__cuda_local_var_242760_615_non_const_sumConductivity_7), __cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_572_non_const_sumCurrents_7), (&__cuda_local_var_242760_615_non_const_sumConductivity_7), __cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_298_non_const_PIdx_7) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_572_non_const_sumCurrents_7), (&__cuda_local_var_242760_615_non_const_sumConductivity_7), __cuda_local_var_242757_645_non_const_v_7, ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 6), ((__cuda_local_var_242757_619_non_const_ModelStates_7) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_298_non_const_PIdx_7])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_666_non_const_sumCurrents_8), (&__cuda_local_var_242760_709_non_const_sumConductivity_8), __cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 0), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[8]), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_666_non_const_sumCurrents_8), (&__cuda_local_var_242760_709_non_const_sumConductivity_8), __cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 2), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[9]), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_666_non_const_sumCurrents_8), (&__cuda_local_var_242760_709_non_const_sumConductivity_8), __cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), ((__cuda_local_var_242757_715_non_const_ModelStates_8)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_666_non_const_sumCurrents_8), (&__cuda_local_var_242760_709_non_const_sumConductivity_8), __cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_666_non_const_sumCurrents_8), (&__cuda_local_var_242760_709_non_const_sumConductivity_8), __cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_345_non_const_PIdx_8) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_666_non_const_sumCurrents_8), (&__cuda_local_var_242760_709_non_const_sumConductivity_8), __cuda_local_var_242757_741_non_const_v_8, ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 6), ((__cuda_local_var_242757_715_non_const_ModelStates_8) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_345_non_const_PIdx_8])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_760_non_const_sumCurrents_9), (&__cuda_local_var_242760_803_non_const_sumConductivity_9), __cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 0), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[8]), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_760_non_const_sumCurrents_9), (&__cuda_local_var_242760_803_non_const_sumConductivity_9), __cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 2), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[9]), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_760_non_const_sumCurrents_9), (&__cuda_local_var_242760_803_non_const_sumConductivity_9), __cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), ((__cuda_local_var_242757_811_non_const_ModelStates_9)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_760_non_const_sumCurrents_9), (&__cuda_local_var_242760_803_non_const_sumConductivity_9), __cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_760_non_const_sumCurrents_9), (&__cuda_local_var_242760_803_non_const_sumConductivity_9), __cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_392_non_const_PIdx_9) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_760_non_const_sumCurrents_9), (&__cuda_local_var_242760_803_non_const_sumConductivity_9), __cuda_local_var_242757_837_non_const_v_9, ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 6), ((__cuda_local_var_242757_811_non_const_ModelStates_9) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_392_non_const_PIdx_9])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_854_non_const_sumCurrents_10), (&__cuda_local_var_242760_899_non_const_sumConductivity_10), __cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 0), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[8]), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_854_non_const_sumCurrents_10), (&__cuda_local_var_242760_899_non_const_sumConductivity_10), __cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 2), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[9]), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_854_non_const_sumCurrents_10), (&__cuda_local_var_242760_899_non_const_sumConductivity_10), __cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), ((__cuda_local_var_242757_910_non_const_ModelStates_10)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_854_non_const_sumCurrents_10), (&__cuda_local_var_242760_899_non_const_sumConductivity_10), __cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_854_non_const_sumCurrents_10), (&__cuda_local_var_242760_899_non_const_sumConductivity_10), __cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_439_non_const_PIdx_10) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_854_non_const_sumCurrents_10), (&__cuda_local_var_242760_899_non_const_sumConductivity_10), __cuda_local_var_242757_937_non_const_v_10, ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 6), ((__cuda_local_var_242757_910_non_const_ModelStates_10) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_439_non_const_PIdx_10])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_952_non_const_sumCurrents_11), (&__cuda_local_var_242760_997_non_const_sumConductivity_11), __cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 0), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[8]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_952_non_const_sumCurrents_11), (&__cuda_local_var_242760_997_non_const_sumConductivity_11), __cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 2), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[9]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_952_non_const_sumCurrents_11), (&__cuda_local_var_242760_997_non_const_sumConductivity_11), __cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), ((__cuda_local_var_242757_1013_non_const_ModelStates_11)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_952_non_const_sumCurrents_11), (&__cuda_local_var_242760_997_non_const_sumConductivity_11), __cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_952_non_const_sumCurrents_11), (&__cuda_local_var_242760_997_non_const_sumConductivity_11), __cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_488_non_const_PIdx_11) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_952_non_const_sumCurrents_11), (&__cuda_local_var_242760_997_non_const_sumConductivity_11), __cuda_local_var_242757_1040_non_const_v_11, ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 6), ((__cuda_local_var_242757_1013_non_const_ModelStates_11) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_488_non_const_PIdx_11])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (0 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_caRdRffS0_S0_fffS0_((&__cuda_local_var_242760_1050_non_const_sumCurrents_12), (&__cuda_local_var_242760_1095_non_const_sumConductivity_12), __cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 0), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[8]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 9));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (1 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_cadRdRffS0_fS0_((&__cuda_local_var_242760_1050_non_const_sumCurrents_12), (&__cuda_local_var_242760_1095_non_const_sumConductivity_12), __cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 2), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[9]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 8));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (2 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z21CuBreakpointModel_kcaRdRffS0_fffff((&__cuda_local_var_242760_1050_non_const_sumCurrents_12), (&__cuda_local_var_242760_1095_non_const_sumConductivity_12), __cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), ((__cuda_local_var_242757_1116_non_const_ModelStates_12)[8]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (3 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kmRdRffS0_fffff((&__cuda_local_var_242760_1050_non_const_sumCurrents_12), (&__cuda_local_var_242760_1095_non_const_sumConductivity_12), __cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (4 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_kvRdRffS0_fffff((&__cuda_local_var_242760_1050_non_const_sumCurrents_12), (&__cuda_local_var_242760_1095_non_const_sumConductivity_12), __cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_537_non_const_PIdx_12) + (5 * ((int)(InMat.N))))])
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z20CuBreakpointModel_naRdRffS0_S0_ffffffffffff((&__cuda_local_var_242760_1050_non_const_sumCurrents_12), (&__cuda_local_var_242760_1095_non_const_sumConductivity_12), __cuda_local_var_242757_1143_non_const_v_12, ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 6), ((__cuda_local_var_242757_1116_non_const_ModelStates_12) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_537_non_const_PIdx_12])))]));
#line 407 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_26_non_const_gModel_1 = ((float)(fdivide((__cuda_local_var_242760_25_non_const_sumCurrentsDv_1 - __cuda_local_var_242760_8_non_const_sumCurrents_1), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_75_non_const_gModel_2 = ((float)(fdivide((__cuda_local_var_242760_119_non_const_sumCurrentsDv_2 - __cuda_local_var_242760_102_non_const_sumCurrents_2), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_124_non_const_gModel_3 = ((float)(fdivide((__cuda_local_var_242760_213_non_const_sumCurrentsDv_3 - __cuda_local_var_242760_196_non_const_sumCurrents_3), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_173_non_const_gModel_4 = ((float)(fdivide((__cuda_local_var_242760_307_non_const_sumCurrentsDv_4 - __cuda_local_var_242760_290_non_const_sumCurrents_4), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_222_non_const_gModel_5 = ((float)(fdivide((__cuda_local_var_242760_401_non_const_sumCurrentsDv_5 - __cuda_local_var_242760_384_non_const_sumCurrents_5), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_271_non_const_gModel_6 = ((float)(fdivide((__cuda_local_var_242760_495_non_const_sumCurrentsDv_6 - __cuda_local_var_242760_478_non_const_sumCurrents_6), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_320_non_const_gModel_7 = ((float)(fdivide((__cuda_local_var_242760_589_non_const_sumCurrentsDv_7 - __cuda_local_var_242760_572_non_const_sumCurrents_7), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_369_non_const_gModel_8 = ((float)(fdivide((__cuda_local_var_242760_683_non_const_sumCurrentsDv_8 - __cuda_local_var_242760_666_non_const_sumCurrents_8), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_418_non_const_gModel_9 = ((float)(fdivide((__cuda_local_var_242760_777_non_const_sumCurrentsDv_9 - __cuda_local_var_242760_760_non_const_sumCurrents_9), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_469_non_const_gModel_10 = ((float)(fdivide((__cuda_local_var_242760_872_non_const_sumCurrentsDv_10 - __cuda_local_var_242760_854_non_const_sumCurrents_10), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_522_non_const_gModel_11 = ((float)(fdivide((__cuda_local_var_242760_970_non_const_sumCurrentsDv_11 - __cuda_local_var_242760_952_non_const_sumCurrents_11), (0.001))));
#line 411 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_575_non_const_gModel_12 = ((float)(fdivide((__cuda_local_var_242760_1068_non_const_sumCurrentsDv_12 - __cuda_local_var_242760_1050_non_const_sumCurrents_12), (0.001))));
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_8_non_const_rhs_1 = (((double)__cuda_local_var_242795_36_non_const_StimCurrent_1) - __cuda_local_var_242760_8_non_const_sumCurrents_1);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_57_non_const_rhs_2 = (((double)__cuda_local_var_242795_85_non_const_StimCurrent_2) - __cuda_local_var_242760_102_non_const_sumCurrents_2);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_106_non_const_rhs_3 = (((double)__cuda_local_var_242795_134_non_const_StimCurrent_3) - __cuda_local_var_242760_196_non_const_sumCurrents_3);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_155_non_const_rhs_4 = (((double)__cuda_local_var_242795_183_non_const_StimCurrent_4) - __cuda_local_var_242760_290_non_const_sumCurrents_4);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_204_non_const_rhs_5 = (((double)__cuda_local_var_242795_232_non_const_StimCurrent_5) - __cuda_local_var_242760_384_non_const_sumCurrents_5);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_253_non_const_rhs_6 = (((double)__cuda_local_var_242795_281_non_const_StimCurrent_6) - __cuda_local_var_242760_478_non_const_sumCurrents_6);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_302_non_const_rhs_7 = (((double)__cuda_local_var_242795_330_non_const_StimCurrent_7) - __cuda_local_var_242760_572_non_const_sumCurrents_7);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_351_non_const_rhs_8 = (((double)__cuda_local_var_242795_379_non_const_StimCurrent_8) - __cuda_local_var_242760_666_non_const_sumCurrents_8);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_400_non_const_rhs_9 = (((double)__cuda_local_var_242795_428_non_const_StimCurrent_9) - __cuda_local_var_242760_760_non_const_sumCurrents_9);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_449_non_const_rhs_10 = (((double)__cuda_local_var_242795_480_non_const_StimCurrent_10) - __cuda_local_var_242760_854_non_const_sumCurrents_10);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_502_non_const_rhs_11 = (((double)__cuda_local_var_242795_533_non_const_StimCurrent_11) - __cuda_local_var_242760_952_non_const_sumCurrents_11);
#line 414 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_555_non_const_rhs_12 = (((double)__cuda_local_var_242795_586_non_const_StimCurrent_12) - __cuda_local_var_242760_1050_non_const_sumCurrents_12);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_15_non_const_D_1 = ((double)(__cuda_local_var_242795_26_non_const_gModel_1 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_16_non_const_PIdx_1])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_15_non_const_D_1 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_64_non_const_D_2 = ((double)(__cuda_local_var_242795_75_non_const_gModel_2 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_63_non_const_PIdx_2])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_64_non_const_D_2 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_113_non_const_D_3 = ((double)(__cuda_local_var_242795_124_non_const_gModel_3 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_110_non_const_PIdx_3])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_113_non_const_D_3 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_162_non_const_D_4 = ((double)(__cuda_local_var_242795_173_non_const_gModel_4 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_157_non_const_PIdx_4])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_162_non_const_D_4 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_211_non_const_D_5 = ((double)(__cuda_local_var_242795_222_non_const_gModel_5 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_204_non_const_PIdx_5])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_211_non_const_D_5 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_260_non_const_D_6 = ((double)(__cuda_local_var_242795_271_non_const_gModel_6 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_251_non_const_PIdx_6])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_260_non_const_D_6 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_309_non_const_D_7 = ((double)(__cuda_local_var_242795_320_non_const_gModel_7 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_298_non_const_PIdx_7])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_309_non_const_D_7 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_358_non_const_D_8 = ((double)(__cuda_local_var_242795_369_non_const_gModel_8 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_345_non_const_PIdx_8])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_358_non_const_D_8 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_407_non_const_D_9 = ((double)(__cuda_local_var_242795_418_non_const_gModel_9 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_392_non_const_PIdx_9])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_407_non_const_D_9 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_457_non_const_D_10 = ((double)(__cuda_local_var_242795_469_non_const_gModel_10 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_439_non_const_PIdx_10])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_457_non_const_D_10 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_510_non_const_D_11 = ((double)(__cuda_local_var_242795_522_non_const_gModel_11 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_488_non_const_PIdx_11])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_510_non_const_D_11 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11)) - 1)]);
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_563_non_const_D_12 = ((double)(__cuda_local_var_242795_575_non_const_gModel_12 + ((float)(fdividef(((double)((cCm)[__cuda_local_var_242754_537_non_const_PIdx_12])), ((double)(__cuda_local_var_242787_12_non_const_dt * (1000.0F))))))));
#line 416 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_563_non_const_D_12 -= ((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12)) - 1)]);
#line 419 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)__cuda_local_var_242754_16_non_const_PIdx_1) == 0)
#line 419 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 420 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242771_16_non_const_parentIndex_1 = ((unsigned short)0U);
#line 421 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 421 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
;
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_92_non_const_dv_1 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_16_non_const_parentIndex_1)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_8_non_const_rhs_1 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1)) - 1)]) * ((double)__cuda_local_var_242757_92_non_const_dv_1));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_188_non_const_dv_2 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_98_non_const_parentIndex_2)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_57_non_const_rhs_2 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2)) - 1)]) * ((double)__cuda_local_var_242757_188_non_const_dv_2));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_284_non_const_dv_3 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_180_non_const_parentIndex_3)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_106_non_const_rhs_3 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3)) - 1)]) * ((double)__cuda_local_var_242757_284_non_const_dv_3));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_380_non_const_dv_4 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_262_non_const_parentIndex_4)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_155_non_const_rhs_4 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4)) - 1)]) * ((double)__cuda_local_var_242757_380_non_const_dv_4));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_476_non_const_dv_5 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_344_non_const_parentIndex_5)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_204_non_const_rhs_5 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5)) - 1)]) * ((double)__cuda_local_var_242757_476_non_const_dv_5));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_572_non_const_dv_6 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_426_non_const_parentIndex_6)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_253_non_const_rhs_6 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6)) - 1)]) * ((double)__cuda_local_var_242757_572_non_const_dv_6));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_668_non_const_dv_7 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_508_non_const_parentIndex_7)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_302_non_const_rhs_7 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7)) - 1)]) * ((double)__cuda_local_var_242757_668_non_const_dv_7));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_764_non_const_dv_8 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_590_non_const_parentIndex_8)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_351_non_const_rhs_8 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8)) - 1)]) * ((double)__cuda_local_var_242757_764_non_const_dv_8));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_860_non_const_dv_9 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_672_non_const_parentIndex_9)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_400_non_const_rhs_9 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9)) - 1)]) * ((double)__cuda_local_var_242757_860_non_const_dv_9));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_962_non_const_dv_10 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_754_non_const_parentIndex_10)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_449_non_const_rhs_10 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10)) - 1)]) * ((double)__cuda_local_var_242757_962_non_const_dv_10));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1065_non_const_dv_11 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_840_non_const_parentIndex_11)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_502_non_const_rhs_11 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11)) - 1)]) * ((double)__cuda_local_var_242757_1065_non_const_dv_11));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1168_non_const_dv_12 += ((__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_926_non_const_parentIndex_12)) - 1)]) - (__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12)) - 1)]));
#line 423 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242795_555_non_const_rhs_12 -= (((cF)[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12)) - 1)]) * ((double)__cuda_local_var_242757_1168_non_const_dv_12));
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1)) - 1)]) = __cuda_local_var_242795_8_non_const_rhs_1;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1)) - 1)]) = __cuda_local_var_242795_15_non_const_D_1;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2)) - 1)]) = __cuda_local_var_242795_57_non_const_rhs_2;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2)) - 1)]) = __cuda_local_var_242795_64_non_const_D_2;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3)) - 1)]) = __cuda_local_var_242795_106_non_const_rhs_3;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3)) - 1)]) = __cuda_local_var_242795_113_non_const_D_3;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4)) - 1)]) = __cuda_local_var_242795_155_non_const_rhs_4;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4)) - 1)]) = __cuda_local_var_242795_162_non_const_D_4;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5)) - 1)]) = __cuda_local_var_242795_204_non_const_rhs_5;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5)) - 1)]) = __cuda_local_var_242795_211_non_const_D_5;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6)) - 1)]) = __cuda_local_var_242795_253_non_const_rhs_6;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6)) - 1)]) = __cuda_local_var_242795_260_non_const_D_6;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7)) - 1)]) = __cuda_local_var_242795_302_non_const_rhs_7;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7)) - 1)]) = __cuda_local_var_242795_309_non_const_D_7;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8)) - 1)]) = __cuda_local_var_242795_351_non_const_rhs_8;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8)) - 1)]) = __cuda_local_var_242795_358_non_const_D_8;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9)) - 1)]) = __cuda_local_var_242795_400_non_const_rhs_9;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9)) - 1)]) = __cuda_local_var_242795_407_non_const_D_9;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10)) - 1)]) = __cuda_local_var_242795_449_non_const_rhs_10;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10)) - 1)]) = __cuda_local_var_242795_457_non_const_D_10;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11)) - 1)]) = __cuda_local_var_242795_502_non_const_rhs_11;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11)) - 1)]) = __cuda_local_var_242795_510_non_const_D_11;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12)) - 1)]) = __cuda_local_var_242795_555_non_const_rhs_12;
#line 427 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12)) - 1)]) = __cuda_local_var_242795_563_non_const_D_12;
#line 428 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_ZN39_INTERNAL_17_CudaStuff_cpp1_ii_1abe6ff811syncthreadsEv();
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_16_non_const_PIdx_1])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_16_non_const_parentIndex_1)) - 1)]) += (((cE)[__cuda_local_var_242771_47_non_const_Eidx_1]) * ((double)__cuda_local_var_242757_92_non_const_dv_1));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_16_non_const_parentIndex_1)) - 1)]) -= ((cE)[__cuda_local_var_242771_47_non_const_Eidx_1]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_63_non_const_PIdx_2])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_98_non_const_parentIndex_2)) - 1)]) += (((cE)[__cuda_local_var_242771_129_non_const_Eidx_2]) * ((double)__cuda_local_var_242757_188_non_const_dv_2));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_98_non_const_parentIndex_2)) - 1)]) -= ((cE)[__cuda_local_var_242771_129_non_const_Eidx_2]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_110_non_const_PIdx_3])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_180_non_const_parentIndex_3)) - 1)]) += (((cE)[__cuda_local_var_242771_211_non_const_Eidx_3]) * ((double)__cuda_local_var_242757_284_non_const_dv_3));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_180_non_const_parentIndex_3)) - 1)]) -= ((cE)[__cuda_local_var_242771_211_non_const_Eidx_3]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_157_non_const_PIdx_4])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_262_non_const_parentIndex_4)) - 1)]) += (((cE)[__cuda_local_var_242771_293_non_const_Eidx_4]) * ((double)__cuda_local_var_242757_380_non_const_dv_4));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_262_non_const_parentIndex_4)) - 1)]) -= ((cE)[__cuda_local_var_242771_293_non_const_Eidx_4]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_204_non_const_PIdx_5])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_344_non_const_parentIndex_5)) - 1)]) += (((cE)[__cuda_local_var_242771_375_non_const_Eidx_5]) * ((double)__cuda_local_var_242757_476_non_const_dv_5));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_344_non_const_parentIndex_5)) - 1)]) -= ((cE)[__cuda_local_var_242771_375_non_const_Eidx_5]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_251_non_const_PIdx_6])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_426_non_const_parentIndex_6)) - 1)]) += (((cE)[__cuda_local_var_242771_457_non_const_Eidx_6]) * ((double)__cuda_local_var_242757_572_non_const_dv_6));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_426_non_const_parentIndex_6)) - 1)]) -= ((cE)[__cuda_local_var_242771_457_non_const_Eidx_6]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_298_non_const_PIdx_7])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_508_non_const_parentIndex_7)) - 1)]) += (((cE)[__cuda_local_var_242771_539_non_const_Eidx_7]) * ((double)__cuda_local_var_242757_668_non_const_dv_7));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_508_non_const_parentIndex_7)) - 1)]) -= ((cE)[__cuda_local_var_242771_539_non_const_Eidx_7]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_345_non_const_PIdx_8])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_590_non_const_parentIndex_8)) - 1)]) += (((cE)[__cuda_local_var_242771_621_non_const_Eidx_8]) * ((double)__cuda_local_var_242757_764_non_const_dv_8));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_590_non_const_parentIndex_8)) - 1)]) -= ((cE)[__cuda_local_var_242771_621_non_const_Eidx_8]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_392_non_const_PIdx_9])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_672_non_const_parentIndex_9)) - 1)]) += (((cE)[__cuda_local_var_242771_703_non_const_Eidx_9]) * ((double)__cuda_local_var_242757_860_non_const_dv_9));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_672_non_const_parentIndex_9)) - 1)]) -= ((cE)[__cuda_local_var_242771_703_non_const_Eidx_9]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_439_non_const_PIdx_10])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_754_non_const_parentIndex_10)) - 1)]) += (((cE)[__cuda_local_var_242771_786_non_const_Eidx_10]) * ((double)__cuda_local_var_242757_962_non_const_dv_10));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_754_non_const_parentIndex_10)) - 1)]) -= ((cE)[__cuda_local_var_242771_786_non_const_Eidx_10]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_488_non_const_PIdx_11])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_840_non_const_parentIndex_11)) - 1)]) += (((cE)[__cuda_local_var_242771_872_non_const_Eidx_11]) * ((double)__cuda_local_var_242757_1065_non_const_dv_11));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_840_non_const_parentIndex_11)) - 1)]) -= ((cE)[__cuda_local_var_242771_872_non_const_Eidx_11]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_537_non_const_PIdx_12])) == 1)
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_926_non_const_parentIndex_12)) - 1)]) += (((cE)[__cuda_local_var_242771_958_non_const_Eidx_12]) * ((double)__cuda_local_var_242757_1168_non_const_dv_12));
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_926_non_const_parentIndex_12)) - 1)]) -= ((cE)[__cuda_local_var_242771_958_non_const_Eidx_12]);
#line 433 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_16_non_const_PIdx_1])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_16_non_const_parentIndex_1)) - 1)]) += (((cE)[__cuda_local_var_242771_47_non_const_Eidx_1]) * ((double)__cuda_local_var_242757_92_non_const_dv_1));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_16_non_const_parentIndex_1)) - 1)]) -= ((cE)[__cuda_local_var_242771_47_non_const_Eidx_1]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_63_non_const_PIdx_2])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_98_non_const_parentIndex_2)) - 1)]) += (((cE)[__cuda_local_var_242771_129_non_const_Eidx_2]) * ((double)__cuda_local_var_242757_188_non_const_dv_2));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_98_non_const_parentIndex_2)) - 1)]) -= ((cE)[__cuda_local_var_242771_129_non_const_Eidx_2]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_110_non_const_PIdx_3])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_180_non_const_parentIndex_3)) - 1)]) += (((cE)[__cuda_local_var_242771_211_non_const_Eidx_3]) * ((double)__cuda_local_var_242757_284_non_const_dv_3));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_180_non_const_parentIndex_3)) - 1)]) -= ((cE)[__cuda_local_var_242771_211_non_const_Eidx_3]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_157_non_const_PIdx_4])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_262_non_const_parentIndex_4)) - 1)]) += (((cE)[__cuda_local_var_242771_293_non_const_Eidx_4]) * ((double)__cuda_local_var_242757_380_non_const_dv_4));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_262_non_const_parentIndex_4)) - 1)]) -= ((cE)[__cuda_local_var_242771_293_non_const_Eidx_4]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_204_non_const_PIdx_5])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_344_non_const_parentIndex_5)) - 1)]) += (((cE)[__cuda_local_var_242771_375_non_const_Eidx_5]) * ((double)__cuda_local_var_242757_476_non_const_dv_5));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_344_non_const_parentIndex_5)) - 1)]) -= ((cE)[__cuda_local_var_242771_375_non_const_Eidx_5]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_251_non_const_PIdx_6])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_426_non_const_parentIndex_6)) - 1)]) += (((cE)[__cuda_local_var_242771_457_non_const_Eidx_6]) * ((double)__cuda_local_var_242757_572_non_const_dv_6));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_426_non_const_parentIndex_6)) - 1)]) -= ((cE)[__cuda_local_var_242771_457_non_const_Eidx_6]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_298_non_const_PIdx_7])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_508_non_const_parentIndex_7)) - 1)]) += (((cE)[__cuda_local_var_242771_539_non_const_Eidx_7]) * ((double)__cuda_local_var_242757_668_non_const_dv_7));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_508_non_const_parentIndex_7)) - 1)]) -= ((cE)[__cuda_local_var_242771_539_non_const_Eidx_7]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_345_non_const_PIdx_8])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_590_non_const_parentIndex_8)) - 1)]) += (((cE)[__cuda_local_var_242771_621_non_const_Eidx_8]) * ((double)__cuda_local_var_242757_764_non_const_dv_8));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_590_non_const_parentIndex_8)) - 1)]) -= ((cE)[__cuda_local_var_242771_621_non_const_Eidx_8]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_392_non_const_PIdx_9])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_672_non_const_parentIndex_9)) - 1)]) += (((cE)[__cuda_local_var_242771_703_non_const_Eidx_9]) * ((double)__cuda_local_var_242757_860_non_const_dv_9));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_672_non_const_parentIndex_9)) - 1)]) -= ((cE)[__cuda_local_var_242771_703_non_const_Eidx_9]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_439_non_const_PIdx_10])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_754_non_const_parentIndex_10)) - 1)]) += (((cE)[__cuda_local_var_242771_786_non_const_Eidx_10]) * ((double)__cuda_local_var_242757_962_non_const_dv_10));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_754_non_const_parentIndex_10)) - 1)]) -= ((cE)[__cuda_local_var_242771_786_non_const_Eidx_10]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_488_non_const_PIdx_11])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_840_non_const_parentIndex_11)) - 1)]) += (((cE)[__cuda_local_var_242771_872_non_const_Eidx_11]) * ((double)__cuda_local_var_242757_1065_non_const_dv_11));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_840_non_const_parentIndex_11)) - 1)]) -= ((cE)[__cuda_local_var_242771_872_non_const_Eidx_11]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if (((int)((cSonNoVec)[__cuda_local_var_242754_537_non_const_PIdx_12])) == 2)
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_15_non_const_bHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_926_non_const_parentIndex_12)) - 1)]) += (((cE)[__cuda_local_var_242771_958_non_const_Eidx_12]) * ((double)__cuda_local_var_242757_1168_non_const_dv_12));
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
(__cuda_local_var_242718_10_non_const_uHP[((((int)(InMat.N)) - ((int)__cuda_local_var_242771_926_non_const_parentIndex_12)) - 1)]) -= ((cE)[__cuda_local_var_242771_958_non_const_Eidx_12]);
#line 436 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 437 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_ZN39_INTERNAL_17_CudaStuff_cpp1_ii_1abe6ff811syncthreadsEv();
#line 438 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z8BeforeLU4HMatPdS0_t(InMat, __cuda_local_var_242718_10_non_const_uHP, __cuda_local_var_242718_15_non_const_bHP, (InMat.Depth));
#line 440 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z5BkSub4HMatPdS0_S0_S0_t(InMat, __cuda_local_var_242750_10_non_const_PX, __cuda_local_var_242750_14_non_const_PF, __cuda_local_var_242718_10_non_const_uHP, __cuda_local_var_242718_15_non_const_bHP, (InMat.LognDepth));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_28_non_const_Vmid_1 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_16_non_const_PIdx_1)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_69_non_const_v_1 += __cuda_local_var_242757_28_non_const_Vmid_1;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_124_non_const_Vmid_2 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_63_non_const_PIdx_2)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_165_non_const_v_2 += __cuda_local_var_242757_124_non_const_Vmid_2;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_220_non_const_Vmid_3 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_110_non_const_PIdx_3)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_261_non_const_v_3 += __cuda_local_var_242757_220_non_const_Vmid_3;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_316_non_const_Vmid_4 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_157_non_const_PIdx_4)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_357_non_const_v_4 += __cuda_local_var_242757_316_non_const_Vmid_4;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_412_non_const_Vmid_5 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_204_non_const_PIdx_5)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_453_non_const_v_5 += __cuda_local_var_242757_412_non_const_Vmid_5;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_508_non_const_Vmid_6 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_251_non_const_PIdx_6)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_549_non_const_v_6 += __cuda_local_var_242757_508_non_const_Vmid_6;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_604_non_const_Vmid_7 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_298_non_const_PIdx_7)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_645_non_const_v_7 += __cuda_local_var_242757_604_non_const_Vmid_7;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_700_non_const_Vmid_8 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_345_non_const_PIdx_8)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_741_non_const_v_8 += __cuda_local_var_242757_700_non_const_Vmid_8;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_796_non_const_Vmid_9 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_392_non_const_PIdx_9)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_837_non_const_v_9 += __cuda_local_var_242757_796_non_const_Vmid_9;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_894_non_const_Vmid_10 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_439_non_const_PIdx_10)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_937_non_const_v_10 += __cuda_local_var_242757_894_non_const_Vmid_10;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_997_non_const_Vmid_11 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_488_non_const_PIdx_11)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1040_non_const_v_11 += __cuda_local_var_242757_997_non_const_Vmid_11;
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1100_non_const_Vmid_12 = ((float)(__cuda_local_var_242750_10_non_const_PX[((((int)(InMat.N)) - ((int)__cuda_local_var_242754_537_non_const_PIdx_12)) - 1)]));
#line 442 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242757_1143_non_const_v_12 += __cuda_local_var_242757_1100_non_const_Vmid_12;
#line 460 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
__cuda_local_var_242749_8_non_const_t += ((0.5) * ((double)__cuda_local_var_242787_12_non_const_dt));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (0 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_caffRfS_fffS_(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 0), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 9));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (1 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z16CuDerivModel_cadffRffS_(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 2), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[9]), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 8));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (2 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z16CuDerivModel_kcaffRffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), ((__cuda_local_var_242757_43_non_const_ModelStates_1)[8]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (3 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_kmffRffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (4 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_kvffRffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_16_non_const_PIdx_1) + (5 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_naffRfS_ffffffffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_69_non_const_v_1, ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 6), ((__cuda_local_var_242757_43_non_const_ModelStates_1) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_16_non_const_PIdx_1])))]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (0 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_caffRfS_fffS_(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 0), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 1), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (0 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (1 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 9));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (1 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z16CuDerivModel_cadffRffS_(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 2), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[9]), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 8));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (2 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z16CuDerivModel_kcaffRffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 3), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (2 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (3 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (4 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (5 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), ((__cuda_local_var_242757_139_non_const_ModelStates_2)[8]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (3 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_kmffRffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 4), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (6 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (7 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (8 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (9 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (10 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (4 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_kvffRffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 5), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (11 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (12 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (13 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (14 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (15 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
if ((cBoolModel)[(((int)__cuda_local_var_242754_63_non_const_PIdx_2) + (5 * ((int)(InMat.N))))])
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
{
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
_Z15CuDerivModel_naffRfS_ffffffffffff(__cuda_local_var_242787_12_non_const_dt, __cuda_local_var_242757_165_non_const_v_2, ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 6), ((__cuda_local_var_242757_139_non_const_ModelStates_2) + 7), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (16 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (17 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (18 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (19 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (20 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (21 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (22 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (23 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (24 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (25 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (26 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]), (ParamsM[(((((int)__cuda_local_var_242747_17_non_const_NeuronID) * ((int)__cuda_local_var_242776_17_non_const_perThreadParamMSize)) + (27 * ((int)(InMat.NComps)))) + ((int)((cSegToComp)[__cuda_local_var_242754_63_non_const_PIdx_2])))]));
#line 465 "C:/pyNeuroGPU_win/NeuroGPU6/CudaStuff.cu"
}
{
}
{
}
{
}
{
}
{
}
{
}
{
}
{