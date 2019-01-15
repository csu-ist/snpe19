//==============================================================================
//
//  Copyright (c) 2017 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   size_t img_width, size_t img_height)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    //zdl::DlSystem::TensorShapeMap tsm;
    //tsm.add("inputdata", {1,500,500,1});
    //snpeBuilder.setInputDimensions(tsm)
    zdl::DlSystem::StringList outSL(1);
    outSL.append("conv_7/BiasAdd");
    snpe = snpeBuilder.setOutputLayers({})     //.setDebugMode(true)
       .setRuntimeProcessor(runtime)
       .setUdlBundle(udlBundle)
       .setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       .setPlatformConfig(platformConfig)
       .build();

    zdl::DlSystem::TensorShapeMap tsm ;//= TensorShapeMap();
	
    const auto& inputNamesOpt = snpe->getInputTensorNames();
    const zdl::DlSystem::StringList& inputNames = *inputNamesOpt;
    const char * inputname_0 = inputNames.at(0);
    
    std::vector<size_t> new_size = {1, img_width, img_height, 1}; 
    zdl::DlSystem::TensorShape input_tensorShape(new_size);

    tsm.add(inputname_0, input_tensorShape);
    
    //snpe = snpeBuilder.setInputDimensions(tsm).build();

    return snpe;
}
