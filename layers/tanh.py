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
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class TanhLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        pass

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Tanh'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        potential_descriptors = []
        for match in matches:
            tanh_op = match['root']
            consumed_nodes = match.consumed_nodes
            potential_descriptors.append(
                TanhLayerResolver.Descriptor('Tanh', str(tanh_op.name), consumed_nodes))
        return potential_descriptors


class TanhLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: TanhLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_neuron_layer(name=descriptor.layer_name,
                                                        func=snpe.modeltools.NEURON_TANH,
                                                        input_name=input_name,
                                                        output_name=output_name,
                                                        a=1.0,
                                                        b=1.0)
