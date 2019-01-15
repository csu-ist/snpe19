#!/usr/bin/env python
#=============================================================================
#
#  Copyright (c) 2015-2016 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#=============================================================================
import numpy as np


from converters.tensorflow.layers.fullyconnected import (
    FullyConnectedLayerResolver,
    FullyConnectedLayerBuilder
)
from converters.tensorflow.layers.convolution import (
    ConvolutionLayerResolver,
    ConvolutionLayerBuilder,
    GroupedConvolutionLayerResolver,
    DilatedConvolutionLayerResolver,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver
)
from converters.tensorflow.layers.concat import (
    ConcatLayerResolver,
    ConcatLayerBuilder
)
from converters.tensorflow.layers.relu import (
    ReluLayerResolver,
    ReluLayerBuilder
)
from converters.tensorflow.layers.relu_min_max import (
    ReluMinMaxLayerResolver,
    ReluMinMaxLayerBuilder
)
from converters.tensorflow.layers.relu6 import (
    Relu6LayerResolver
)
from converters.tensorflow.layers.sigmoid import (
    SigmoidLayerResolver,
    SigmoidLayerBuilder
)
from converters.tensorflow.layers.tanh import (
    TanhLayerResolver,
    TanhLayerBuilder
)
from converters.tensorflow.layers.softmax import (
    SoftmaxLayerResolver,
    SoftmaxLayerBuilder
)
from converters.tensorflow.layers.lrn import (
    LrnLayerResolver,
    LrnLayerBuilder
)
from converters.tensorflow.layers.deconvolution import (
    DeConvolutionOptimizedLayerResolver,
    DeConvolutionLayerBuilder
)
from converters.tensorflow.layers.batchnorm import (
    BatchNormLayerResolver,
    UnscaledBatchNormLayerResolver,
    ScaledBatchNormLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    BatchNormLayerBuilder,
    FusedBatchNormNormLayerResolver,
    GenericBatchNormLayerResolver
)
from converters.tensorflow.layers.pooling import (
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    PoolingLayerBuilder
)
from converters.tensorflow.layers.eltwise import (
    EltWiseSumLayerResolver,
    EltWiseSumLayerBuilder,
    EltWiseSubLayerResolver,
    EltWiseSubLayerBuilder,
    EltWiseMulLayerResolver,
    EltWiseMulLayerBuilder,
    EltWiseMaxLayerResolver,
    EltWiseMaxLayerBuilder
)

from converters.tensorflow.layers.add_n import (
    AddNLayerResolver,
    AddNLayerBuilder
)

from converters.tensorflow.layers.slice import (
    SliceLayerResolver,
    SliceLayerBuilder
)

from converters.tensorflow.layers.prelu import (
    PReLuLayerResolver,
    PReLuLayerBuilder
)

from converters.tensorflow.layers.reshape import (
    ReshapeLayerResolver,
    ReshapeLayerBuilder
)

from converters.tensorflow.layers.resize import (
    ResizeNearestNeighborLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeLayerBuilder
)

from converters.tensorflow.layers.lstm import (
    LstmLayerResolver,
    LstmLayerBuilder
)
from converters.tensorflow.layers.ignored_patterns import (
    IgnoredLayersResolver,
    IgnoredLayersBuilder
)

from converters.tensorflow.layers.fill import (
    FillLayerResolver,
    FillLayerBuilder
)

from converters.tensorflow.layers.ssd import (
    SSDDecoderResolver,
    SSDDecoderLayersBuilder,
    SSDNmsResolver,
    SSDNmsLayersBuilder,
    SSDAnchorGeneratorResolver,
)

from converters.tensorflow.layers.crop import (
    CropLayerResolver,
    CropLayerBuilder
)

from converters.tensorflow.layers.constant import (
    ConstantLayerResolver,
    ConstantLayerBuilder
)

from converters.tensorflow.layers.pad import (
    PadLayerResolver,
    PadLayerBuilder
)

from converters.tensorflow.layers.strided_slice import (
    StridedSliceLayerResolver,
    StridedSliceLayerBuilder
)

from converters.tensorflow.layers.permute import (
    PermuteLayerResolver,
    PermuteLayerBuilder
)

from converters.tensorflow.layers.argmax import (
    ArgMaxLayerResolver,
    ArgMaxLayerBuilder
)

from converters.tensorflow.layers.channel_shuffle import (
    ChannelShuffleLayerResolver,
    ChannelShuffleLayerBuilder
)

from converters.tensorflow.layers.elu import (
    EluLayerResolver,
    EluLayerBuilder
)

from converters.tensorflow.layers.mean import (
    MeanLayerResolver,
    MeanLayerBuilder
)

from converters.tensorflow.common import (
    LayerDescriptor,
    LayerResolver,
    LayerBuilder
)

layer_resolvers = [
    IgnoredLayersResolver,
    SSDAnchorGeneratorResolver,
    SSDNmsResolver,
    ConvolutionLayerResolver,
    ConcatLayerResolver,
    FullyConnectedLayerResolver,
    ReluLayerResolver,
    Relu6LayerResolver,
    SigmoidLayerResolver,
    TanhLayerResolver,
    AvgPoolingLayerResolver,
    MaxPoolingLayerResolver,
    SoftmaxLayerResolver,
    LrnLayerResolver,
    DeConvolutionOptimizedLayerResolver,
    EltWiseSumLayerResolver,
    EltWiseSubLayerResolver,
    EltWiseMulLayerResolver,
    EltWiseMaxLayerResolver,
    UnscaledBatchNormLayerResolver,
    ScaledBatchNormLayerResolver,
    BatchNormWithGlobalNormLayerResolver,
    GenericBatchNormLayerResolver,
    GroupedConvolutionLayerResolver,
    SliceLayerResolver,
    PReLuLayerResolver,
    DilatedConvolutionLayerResolver,
    ReshapeLayerResolver,
    ResizeBilinearLayerResolver,
    ResizeNearestNeighborLayerResolver,
    DepthwiseConvolutionLayerResolver,
    DilatedDepthwiseConvolutionLayerResolver,
    AddNLayerResolver,
    LstmLayerResolver,
    FillLayerResolver,
    SSDDecoderResolver,
    CropLayerResolver,
    FusedBatchNormNormLayerResolver,
    PadLayerResolver,
    StridedSliceLayerResolver,
    PermuteLayerResolver,
    ArgMaxLayerResolver,
    ChannelShuffleLayerResolver,
    EluLayerResolver,
    MeanLayerResolver
]
"""
type: list[type(LayerResolver)]
"""

layer_builders = {
    BatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    BatchNormWithGlobalNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    GenericBatchNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    ConcatLayerResolver.Descriptor: ConcatLayerBuilder,
    ConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    DeConvolutionOptimizedLayerResolver.Descriptor: DeConvolutionLayerBuilder,
    EltWiseMaxLayerResolver.Descriptor: EltWiseMaxLayerBuilder,
    EltWiseMulLayerResolver.Descriptor: EltWiseMulLayerBuilder,
    EltWiseSumLayerResolver.Descriptor: EltWiseSumLayerBuilder,
    EltWiseSubLayerResolver.Descriptor: EltWiseSubLayerBuilder,
    AddNLayerResolver.Descriptor: AddNLayerBuilder,
    FullyConnectedLayerResolver.Descriptor: FullyConnectedLayerBuilder,
    LrnLayerResolver.Descriptor: LrnLayerBuilder,
    ReluLayerResolver.Descriptor: ReluLayerBuilder,
    Relu6LayerResolver.Descriptor: ReluMinMaxLayerBuilder,
    SigmoidLayerResolver.Descriptor: SigmoidLayerBuilder,
    SoftmaxLayerResolver.Descriptor: SoftmaxLayerBuilder,
    TanhLayerResolver.Descriptor: TanhLayerBuilder,
    AvgPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    MaxPoolingLayerResolver.Descriptor: PoolingLayerBuilder,
    GroupedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    SliceLayerResolver.Descriptor: SliceLayerBuilder,
    PReLuLayerResolver.Descriptor: PReLuLayerBuilder,
    DilatedConvolutionLayerResolver.Descriptor: ConvolutionLayerBuilder,
    ReshapeLayerResolver.Descriptor: ReshapeLayerBuilder,
    ResizeBilinearLayerResolver.Descriptor: ResizeLayerBuilder,
    ResizeNearestNeighborLayerResolver.Descriptor: ResizeLayerBuilder,
    LstmLayerResolver.UnrolledTimeStepDescriptor: LstmLayerBuilder,
    LstmLayerResolver.StateDescriptor: LstmLayerBuilder,
    IgnoredLayersResolver.Descriptor: IgnoredLayersBuilder,
    FillLayerResolver.Descriptor: FillLayerBuilder,
    SSDDecoderResolver.Descriptor: SSDDecoderLayersBuilder,
    CropLayerResolver.Descriptor: CropLayerBuilder,
    SSDNmsResolver.Descriptor: SSDNmsLayersBuilder,
    ConstantLayerResolver.Descriptor: ConstantLayerBuilder,
    FusedBatchNormNormLayerResolver.Descriptor: BatchNormLayerBuilder,
    PadLayerResolver.Descriptor: PadLayerBuilder,
    StridedSliceLayerResolver.Descriptor: StridedSliceLayerBuilder,
    PermuteLayerResolver.Descriptor: PermuteLayerBuilder,
    ArgMaxLayerResolver.Descriptor: ArgMaxLayerBuilder,
    ChannelShuffleLayerResolver.Descriptor: ChannelShuffleLayerBuilder,
    EluLayerResolver.Descriptor: EluLayerBuilder,
    MeanLayerResolver.Descriptor: MeanLayerBuilder
}
"""
type: dict[type(LayerDescriptor), type(LayerBuilder)]
"""
