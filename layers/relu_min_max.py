#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2015-2016 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import snpe

from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder


class ReluMinMaxLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, layer_type, name, nodes, min_clamp=0, max_clamp=0):
            super(ReluMinMaxLayerResolver.Descriptor, self).__init__(layer_type, name, nodes)
            self.min_clamp = min_clamp
            self.max_clamp = max_clamp


class ReluMinMaxLayerBuilder(LayerBuilder):

    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReluLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_neuron_layer(name=descriptor.layer_name,
                                                        func=snpe.modeltools.NEURON_RELU_MIN_MAX,
                                                        input_name=input_name,
                                                        output_name=output_name,
                                                        min_clamp=descriptor.min_clamp,
                                                        max_clamp=descriptor.max_clamp)
