#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2018 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
from converters import code_to_message
from converters.tensorflow.util import ConverterError
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.graph_matcher import (
    ConverterSequenceNode,
    NonConsumableConverterSequenceNode,
    GraphSequence
)
import numpy as np
import snpe


class PadLayerResolver(LayerResolver, object):

    class Descriptor(LayerDescriptor):
        def __init__(self, name, nodes, paddings, mode, constant_values, output_names=None):
            super(PadLayerResolver.Descriptor, self).__init__('Pad', name, nodes, output_names=output_names)
            self.paddings = paddings
            self.mode = mode
            self.constant_values = constant_values

    def __init__(self):
        self.sequence_with_zero_padding = GraphSequence([
            ConverterSequenceNode('root', ['Pad', 'PadV2']),
            ConverterSequenceNode('paddings', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence_with_zero_padding.set_inputs('root', ['input', 'paddings'])
        self.sequence_with_zero_padding.set_outputs(['root'])

        self.sequence_with_const_padding = GraphSequence([
            ConverterSequenceNode('root', ['Pad', 'PadV2']),
            ConverterSequenceNode('paddings', ['Const']),
            ConverterSequenceNode('const_values', ['Const']),
            NonConsumableConverterSequenceNode('input', ['?']),
        ])
        self.sequence_with_const_padding.set_inputs('root', ['input', 'paddings', 'const_values'])
        self.sequence_with_const_padding.set_outputs(['root'])

        self.sequences = [self.sequence_with_zero_padding, self.sequence_with_const_padding]

    def resolve_layer(self, graph_matcher, graph_helper):
        descriptors = []
        for sequence in self.sequences:
            for match in graph_matcher.match_sequence(sequence):
                pad_op = match['root']
                input_op = match['input']
                paddings_op = match['paddings']

                paddings_tensor = graph_helper.evaluate_tensor_output(paddings_op.outputs[0])
                paddings_shape = graph_helper.get_op_output_shape(paddings_op)

                input_rank = len(graph_helper.get_op_output_shape(input_op))
                if [input_rank, 2] != paddings_shape:
                    raise ConverterError(code_to_message.get_message(
                        'ERROR_TF_PAD_INVALUD_PADDINGS')(str([input_rank, 2]), str(paddings_shape)))

                if 'const_values' in match:
                    const_values_op = match['const_values']
                    const_values = graph_helper.evaluate_tensor_output(const_values_op.outputs[0])
                else:
                    const_values = 0.0

                if not np.isscalar(const_values):
                    raise ConverterError(code_to_message.get_message('ERROR_TF_PAD_CONSTANT_NOT_SCALAR'))

                consumed_nodes = match.consumed_nodes
                pad_descriptor = PadLayerResolver.Descriptor(
                    str(pad_op.name), consumed_nodes, paddings_tensor,
                    snpe.modeltools.PADDING_CONSTANT, const_values,
                    output_names=[str(pad_op.outputs[0].name)])
                descriptors.extend([pad_descriptor])

        return descriptors


class PadLayerBuilder(LayerBuilder):
    def build_layer(self, converter_context, descriptor, input_descriptors, output_descriptors):
        """
        :type input_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type output_descriptors: [converters.tensorflow.common.LayerDescriptor]
        :type converter_context: converters.tensorflow.converter.ConverterContext
        :type descriptor: PadLayerResolver.Descriptor
        :rtype: int
        """
        input_name = self.get_input_name(converter_context, descriptor, input_descriptors)
        output_name = descriptor.output_names[0]
        return converter_context.model.add_pad_layer(name=descriptor.layer_name,
                                                     input_name=input_name,
                                                     paddings=descriptor.paddings.tolist(),
                                                     mode=descriptor.mode,
                                                     constant_values=float(descriptor.constant_values),
                                                     output_name=output_name)
