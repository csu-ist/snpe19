//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifdef ANDROID

#include <stdlib.h>
#include <time.h>
#include <string>
#include <iostream>
#include <vector>

#include "CreateGLBuffer.hpp"

CreateGLBuffer::CreateGLBuffer(){
    this->createGLContext();
}

CreateGLBuffer::~CreateGLBuffer()
{
}

void CreateGLBuffer::createGLContext()
{
    const EGLint attribListWindow[] =
    {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };
    const EGLint attribListPbuffer[] =
    {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 16,
        EGL_STENCIL_SIZE, 0,
        EGL_NONE
    };
    const EGLint srfPbufferAttr[] =
    {
        EGL_WIDTH, 1024,
        EGL_HEIGHT, 1024,
        EGL_COLORSPACE, GL_RGB,
        EGL_TEXTURE_FORMAT, EGL_TEXTURE_RGB,
        EGL_TEXTURE_TARGET, EGL_TEXTURE_2D,
        EGL_LARGEST_PBUFFER, EGL_TRUE,
        EGL_NONE
    };
    static const EGLint gl_context_attribs[] =
    {
        EGL_CONTEXT_CLIENT_VERSION, 3,
        EGL_NONE
    };
    EGLDisplay eglDisplay = 0;
    EGLConfig eglConfigWindow = 0;
    EGLConfig eglConfigPbuffer = 0;
    EGLSurface eglSurfaceWindow = 0;
    EGLSurface eglSurfacePbuffer = 0;
    EGLContext eglContext = 0;
    EGLint iMajorVersion, iMinorVersion;
    int iConfigs;
    eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(eglDisplay, &iMajorVersion, &iMinorVersion);
    eglChooseConfig(eglDisplay, attribListWindow,
                    &eglConfigWindow, 1, &iConfigs);
    eglContext = eglCreateContext(eglDisplay,
                                  eglConfigWindow, EGL_NO_CONTEXT, gl_context_attribs);
    eglSurfaceWindow = eglGetCurrentSurface(EGL_DRAW);
    eglSurfacePbuffer = eglCreatePbufferSurface(eglDisplay,
                        eglConfigPbuffer,srfPbufferAttr);

    eglMakeCurrent(eglDisplay, eglSurfacePbuffer, eglSurfacePbuffer, eglContext);
}

GLuint CreateGLBuffer::convertImage2GLBuffer(const std::vector<std::string>& fileLines)
{
   std::cout << "Processing DNN Input: " << std::endl;
   std::vector<uint8_t> inputVec;
   for(size_t i = 0; i < fileLines.size(); ++i)
   {
      std::string fileLine(fileLines[i]);
      // treat each line as a space-separated list of input files
      std::vector<std::string> filePaths;
      split(filePaths, fileLine, ' ');
      std::string filePath(filePaths[0]);
      std::cout << "\t" << i + 1 << ") " << filePath << std::endl;
      loadByteDataFileBatched(filePath, inputVec, i);
   }
   size_t length = inputVec.size()*sizeof(uint8_t);
   GLuint userBuffers;
   glGenBuffers(1, &userBuffers);
   glBindBuffer(GL_SHADER_STORAGE_BUFFER, userBuffers);
   glBufferData(GL_SHADER_STORAGE_BUFFER, length, inputVec.data(), GL_STREAM_DRAW);

   return userBuffers;
}

void CreateGLBuffer::SetGPUPlatformConfig(zdl::DlSystem::PlatformConfig& platformConfig)
{
    void* glcontext = eglGetCurrentContext();
    void* gldisplay = eglGetCurrentDisplay();
    zdl::DlSystem::UserGLConfig userGLConfig;
    userGLConfig.userGLContext = glcontext;
    userGLConfig.userGLDisplay = gldisplay;
    zdl::DlSystem::UserGpuConfig userGpuConfig;
    userGpuConfig.userGLConfig = userGLConfig;
    platformConfig.setUserGpuConfig(userGpuConfig);
}

#endif
