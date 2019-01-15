//==============================================================================
//
//  Copyright (c) 2017-2018 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include <cerrno>

#include "Util.hpp"

#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/TensorShape.hpp"

size_t calcSizeFromDims(const zdl::DlSystem::Dimension *dims, size_t rank, size_t elementSize)
{
   if (rank == 0) return 0;
   size_t size = elementSize;
   while (rank--) size *= *dims++;
   return size;
}


bool EnsureDirectory(const std::string& dir)
{
   auto i = dir.find_last_of('/');
   std::string prefix = dir.substr(0, i);

   if (dir.empty() || dir == "." || dir == "..")
   {
      return true;
   }

   if (i != std::string::npos && !EnsureDirectory(prefix))
   {
      return false;
   }

   int rc = mkdir(dir.c_str(),  S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
   if (rc == -1 && errno != EEXIST)
   {
      return false;
   }
   else
   {
      struct stat st;
      if (stat(dir.c_str(), &st) == -1)
      {
         return false;
      }

      return S_ISDIR(st.st_mode);
   }
}

std::vector<float> loadFloatDataFile(const std::string& inputFile)
{
    std::vector<float> vec;
    loadByteDataFile(inputFile, vec);
    return vec;
}

std::vector<unsigned char> loadByteDataFile(const std::string& inputFile)
{
   std::vector<unsigned char> vec;
   loadByteDataFile(inputFile, vec);
   return vec;
}

template<typename T>
void loadByteDataFile(const std::string& inputFile, std::vector<T>& loadVector)
{
   std::ifstream in(inputFile, std::ifstream::binary);
   if (!in.is_open() || !in.good())
   {
      std::cerr << "Failed to open input file: " << inputFile << "\n";
   }

   in.seekg(0, in.end);
   size_t length = in.tellg();
   in.seekg(0, in.beg);

   if (loadVector.size() == 0) {
      loadVector.resize(length / sizeof(T));
   } else if (loadVector.size() < length) {
      std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
   }

   if (!in.read(reinterpret_cast<char*>(&loadVector[0]), length))
   {
      std::cerr << "Failed to read the contents of: " << inputFile << "\n";
   }
}

std::vector<unsigned char> loadByteDataFileBatched(const std::string& inputFile)
{
   std::vector<unsigned char> vec;
   size_t offset=0;
   loadByteDataFileBatched(inputFile, vec, offset);
   return vec;
}

template<typename T>
void loadByteDataFileBatched(const std::string& inputFile, std::vector<T>& loadVector, size_t offset)
{
    std::ifstream in(inputFile, std::ifstream::binary);
    if (!in.is_open() || !in.good())
    {
        std::cerr << "Failed to open input file: " << inputFile << "\n";
    }

    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg(0, in.beg);

    if (loadVector.size() == 0) {
        loadVector.resize(length / sizeof(T));
    } else if (loadVector.size() < length) {
        std::cerr << "Vector is not large enough to hold data of input file: " << inputFile << "\n";
    }

    loadVector.resize( (offset+1) * length / sizeof(T) );

    if (!in.read( reinterpret_cast<char*> (&loadVector[offset * length/ sizeof(T) ]), length) )
    {
        std::cerr << "Failed to read the contents of: " << inputFile << "\n";
    }
}

void SaveITensorBatched(const std::string& path, const zdl::DlSystem::ITensor* tensor, size_t batchIndex, size_t batchChunk)
{
   if(batchChunk == 0)
      batchChunk = tensor->getSize();
   // Create the directory path if it does not exist
   auto idx = path.find_last_of('/');
   if (idx != std::string::npos)
   {
      std::string dir = path.substr(0, idx);
      if (!EnsureDirectory(dir))
      {
         std::cerr << "Failed to create output directory: " << dir << ": "
                   << std::strerror(errno) << "\n";
         std::exit(EXIT_FAILURE);
      }
   }

   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }

   for ( auto it = tensor->cbegin() + batchIndex * batchChunk; it != tensor->cbegin() + (batchIndex+1) * batchChunk; ++it )
   {
      float f = *it;
      if (!os.write(reinterpret_cast<char*>(&f), sizeof(float)))
      {
         std::cerr << "Failed to write data to: " << path << "\n";
         std::exit(EXIT_FAILURE);
      }
   }
}

void SaveUserBufferBatched(const std::string& path, const std::vector<uint8_t>& buffer, size_t batchIndex, size_t batchChunk)
{
   if(batchChunk == 0)
      batchChunk = buffer.size();
   // Create the directory path if it does not exist
   auto idx = path.find_last_of('/');
   if (idx != std::string::npos)
   {
      std::string dir = path.substr(0, idx);
      if (!EnsureDirectory(dir))
      {
         std::cerr << "Failed to create output directory: " << dir << ": "
                   << std::strerror(errno) << "\n";
         std::exit(EXIT_FAILURE);
      }
   }

   std::ofstream os(path, std::ofstream::binary);
   if (!os)
   {
      std::cerr << "Failed to open output file for writing: " << path << "\n";
      std::exit(EXIT_FAILURE);
   }

   for ( auto it = buffer.begin() + batchIndex * batchChunk; it != buffer.begin() + (batchIndex+1) * batchChunk; ++it )
   {
      uint8_t u = *it;
      if(!os.write((char*)(&u), sizeof(uint8_t)))
      {
        std::cerr << "Failed to write data to: " << path << "\n";
        std::exit(EXIT_FAILURE);
      }
   }
}
