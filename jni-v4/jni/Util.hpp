//==============================================================================
//
//  Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>
#include <sstream>

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorShape.hpp"

template <typename Container> Container& split(Container& result, const typename Container::value_type & s, typename Container::value_type::value_type delimiter )
{
  result.clear();
  std::istringstream ss( s );
  while (!ss.eof())
  {
    typename Container::value_type field;
    getline( ss, field, delimiter );
    if (field.empty()) continue;
    result.push_back( field );
  }
  return result;
}

size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank, size_t elementSize);

std::vector<float> loadFloatDataFile(const std::string& inputFile);
std::vector<unsigned char> loadByteDataFile(const std::string& inputFile);
template<typename T> void loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector);

std::vector<unsigned char> loadByteDataFileBatched(const std::string& inputFile);
template<typename T> void loadByteDataFileBatched(const std::string& inputFile, std::vector<T>& loadVector, size_t offset=1);

void SaveITensorBatched(const std::string& path, const zdl::DlSystem::ITensor* tensor, size_t batchIndex=0, size_t batchChunk=0);
void SaveUserBufferBatched(const std::string& path, const std::vector<uint8_t>& buffer, size_t batchIndex=0, size_t batchChunk=0);
bool EnsureDirectory(const std::string& dir);

#endif

