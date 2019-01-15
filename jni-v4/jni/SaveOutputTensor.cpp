//==============================================================================
//
//  Copyright (c) 2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <algorithm>
#include <sstream>
#include <unordered_map>

#include "SaveOutputTensor.hpp"
#include "Util.hpp"

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"

// Print the results to raw files
// ITensor
void saveOutput (zdl::DlSystem::TensorMap outputTensorMap,
                 const std::string& outputDir,
                 int num,
                 size_t batchSize)
{
    // Get all output tensor names from the network
    zdl::DlSystem::StringList tensorNames = outputTensorMap.getTensorNames();

    // Iterate through the output Tensor map, and print each output layer name to a raw file
    std::for_each( tensorNames.begin(), tensorNames.end(), [&](const char* name)
    {
        // Split the batched output tensor and save the results
        for(size_t i=0; i<batchSize; i++) {
            std::ostringstream path;
            path << outputDir << "/"
                 << "Result_" << num + i << "/"
                 << name << ".raw";
            auto tensorPtr = outputTensorMap.getTensor(name);
            size_t batchChunk = tensorPtr->getSize() / batchSize;

            SaveITensorBatched(path.str(), tensorPtr, i, batchChunk);
        }
    });
}

// Execute the network on an input user buffer map and print results to raw files
void saveOutput (zdl::DlSystem::UserBufferMap& outputMap,
                 std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
                 const std::string& outputDir,
                 int num,
                 size_t batchSize)
{
   // Get all output buffer names from the network
   const zdl::DlSystem::StringList& outputBufferNames = outputMap.getUserBufferNames();

   // Iterate through output buffers and print each output to a raw file
   std::for_each(outputBufferNames.begin(), outputBufferNames.end(), [&](const char* name)
   {
       for(size_t i=0; i<batchSize; i++) {
           std::ostringstream path;
           path << outputDir << "/"
                << "Result_" << num + i << "/"
                << name << ".raw";
           auto bufferPtr = outputMap.getUserBuffer(name);
           size_t batchChunk = bufferPtr->getSize() / batchSize;
           SaveUserBufferBatched(path.str(),applicationOutputBuffers.at(name), i, batchChunk);
       }
   });
}
