#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2015-2017 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from .fullyconnected import FullyConnectedLayerResolver
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    GraphSequence
)


class ReshapeLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes):
            super(ReshapeLayerResolver.Descriptor, self).__init__('Reshape', name, nodes)

    def __init__(self):
        self.sequence = GraphSequence([ConverterSequenceNode('root', ['Reshape', 'Squeeze', 'ExpandDims'])])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        matches = graph_matcher.match_sequence(self.sequence)
        if len(matches) == 0:
            return []
        descriptors = []
        for match in matches:
            reshape_op = match['root']
            consumed_nodes = match.consumed_nodes
            descriptors.append(
                ReshapeLayerResolver.Descriptor(str(reshape_op.name), consumed_nodes))
        return descriptors


class ReshapeLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: ReshapeLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors[:1])
        output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.child_ops[0])
        output_shape = output_shape[-4:] if len(output_shape) > 4 else output_shape
        return converter_context.model.add_reshape_layer(descriptor.output_names[0],
                                                         output_shape,
                                                         input_name,
                                                         descriptor.output_names[0])

    def transform_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        fc_outputs = [d for d in output_descriptors if isinstance(d, FullyConnectedLayerResolver.Descriptor)]
        if len(output_descriptors) == 1 and fc_outputs == output_descriptors:
            converter_context.merge_descriptors(descriptor, fc_outputs[0])
            return

        non_ignored_inputs = [d for d in input_descriptors if not d.is_ignored]
        if len(non_ignored_inputs) == 1:
            tensors = converter_context.get_output_tensors_between(non_ignored_inputs[0], descriptor)
            input_shape = converter_context.graph_helper.get_op_output_shape(tensors[0].op)
            output_shape = converter_context.graph_helper.get_op_output_shape(descriptor.child_ops[0])
            if input_shape == output_shape:
                converter_context.merge_descriptors(descriptor, non_ignored_inputs[0])
        elif len(non_ignored_inputs) == 0:
            descriptor.set_ignored(True)
