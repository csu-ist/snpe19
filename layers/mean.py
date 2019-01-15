#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import numpy as np
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)


class MeanLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, axes, keep_dims, output_names=None):
            super(MeanLayerResolver.Descriptor, self).__init__('Mean', name, nodes, output_names=output_names)
            self.axes = axes
            self.keep_dims = keep_dims

    def __init__(self):
        self.sequence = GraphSequence([
            ConverterSequenceNode('root', ['Mean']),
            ConverterSequenceNode('reduction_indices', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence.set_inputs('root', ['input', 'reduction_indices'])
        self.sequence.set_outputs(['root'])

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for match in graph_matcher.match_sequence(self.sequence):
            mean_op = match['root']
            input_op = match['input']
            reduction_indices_op = match['reduction_indices']

            axes = graph_helper.evaluate_tensor_output(reduction_indices_op.outputs[0])
            keep_dims = bool(mean_op.get_attr('keep_dims'))

            input_shape = graph_helper.get_op_output_shape(input_op)
            input_rank = len(input_shape)

            axes = [axes] if np.isscalar(axes) else axes.tolist()
            for i in range(len(axes)):
                axes[i] = int(axes[i])
                if axes[i] < 0:
                    axes[i] += input_rank

            mean_descriptor = MeanLayerResolver.Descriptor(
                str(mean_op.name), match.consumed_nodes, axes, keep_dims,
                output_names=[str(mean_op.outputs[0].name)])
            descriptors.extend([mean_descriptor])

        return descriptors


class MeanLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: MeanLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_mean_layer(name=descriptor.layer_name,
                                                      input_name=input_name,
                                                      output_name=output_name,
                                                      axes=descriptor.axes,
                                                      keep_dims=descriptor.keep_dims)
