#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2015-2016 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import snpe

from converters import code_to_message
from converters.tensorflow.common import LayerDescriptor, LayerResolver, LayerBuilder
from converters.tensorflow.util import ConverterError
from converters.tensorflow.graph_matcher import(
    ConverterSequenceNode,
    GraphSequence,
    NonConsumableConverterSequenceNode
)


class PReLuLayerResolver(LayerResolver, object):
    class Descriptor(LayerDescriptor):
        def __init__(self, name, operations, coefficients, output_names):
            super(PReLuLayerResolver.Descriptor, self).__init__('PReLU', name, operations,
                                                                output_names=output_names)
            self.coefficients = coefficients

    def __init__(self):
        sequence_prelu = GraphSequence([
            ConverterSequenceNode('a', ['Relu']),
            ConverterSequenceNode('b', ['Abs']),
            ConverterSequenceNode('c', ['Sub']),
            ConverterSequenceNode('d', ['Mul']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            ConverterSequenceNode('unknown', ['?']),
            ConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu.set_inputs('a', ['inputs'])
        sequence_prelu.set_inputs('b', ['inputs'])
        sequence_prelu.set_inputs('c', ['inputs', 'b'])
        sequence_prelu.set_inputs('d', ['alphas', 'c'])
        sequence_prelu.set_inputs('e', ['d', 'unknown'])
        sequence_prelu.set_inputs('f', ['a', 'e'])
        sequence_prelu.set_outputs(['f'])

        sequence_prelu_negative = GraphSequence([
            ConverterSequenceNode('a', ['Relu']),
            ConverterSequenceNode('b', ['Neg']),
            ConverterSequenceNode('c', ['Neg']),
            ConverterSequenceNode('d', ['Relu']),
            ConverterSequenceNode('e', ['Mul']),
            ConverterSequenceNode('f', ['Add']),  # output
            ConverterSequenceNode('alphas', ['?']),
            NonConsumableConverterSequenceNode('inputs', ['?'])
        ])
        sequence_prelu_negative.set_inputs('a', ['inputs'])
        sequence_prelu_negative.set_inputs('b', ['inputs'])
        sequence_prelu_negative.set_inputs('c', ['alphas'])
        sequence_prelu_negative.set_inputs('d', ['b'])
        sequence_prelu_negative.set_inputs('e', ['d', 'c'])
        sequence_prelu_negative.set_inputs('f', ['a', 'e'])
        sequence_prelu_negative.set_outputs(['f'])

        self.sequences = [sequence_prelu, sequence_prelu_negative]

    def resolve_layer(self, graph_matcher, graph_helper):
        potential_descriptors = []
        for sequence in self.sequences:
            matches = graph_matcher.match_sequence(sequence)
            for match in matches:
                coefficients = match['alphas']
                add_op = match['f']
                if coefficients.type not in ['Identity', 'Const']:
                    raise ConverterError(code_to_message.get_message('ERROR_TF_RESOLVE_PRELU_COEFF'))

                output_op_nodes_names = [str(match[node.identifier].outputs[0].name) for node in sequence.output_nodes]
                consumed_nodes = match.consumed_nodes
                potential_descriptors.append(
                    PReLuLayerResolver.Descriptor(str(add_op.name), consumed_nodes,
                                                  graph_helper.evaluate_tensor_output(coefficients.outputs[0]),
                                                  output_names=output_op_nodes_names))
        return potential_descriptors


class PReLuLayerBuilder(LayerBuilder):
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
        return converter_context.model.add_prelu_layer(name=descriptor.layer_name,
                                                       coeff=descriptor.coefficients.flatten().tolist(),
                                                       input_name=input_name,
                                                       output_name=output_name)
