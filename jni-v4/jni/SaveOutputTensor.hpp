//==============================================================================
//
//  Copyright (c) 2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SAVEOUTPUTTENSOR_H
#define SAVEOUTPUTTENSOR_H

#include <string>
#include <unordered_map>
#include <vector>

#include "SNPE/SNPE.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/UserBufferMap.hpp"

// Save output implementation of ITensor
void saveOutput (zdl::DlSystem::TensorMap outputTensorMap,
                 const std::string& outputDir,
                 int num,
                 size_t batchSize=1);

// Save output USERBUFFER
void saveOutput (zdl::DlSystem::UserBufferMap& outputMap,
                 std::unordered_map<std::string,std::vector<uint8_t>>& applicationOutputBuffers,
                 const std::string& outputDir,
                 int num,
                 size_t batchSize=1);



#endif
