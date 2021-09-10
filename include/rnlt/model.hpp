#pragma once

#include <torch/torch.h>

namespace rnlt {

static torch::nn::Conv2dOptions conv_options(int64_t ic, int64_t oc, int64_t k,
                                      int64_t s, int64_t p=0, bool bias=false) {
    torch::nn::Conv2dOptions options(ic, oc, k);
    options.stride() = s, options.padding() = p, options.bias() = bias;
    return options;
}

/// Basic block of ResNet structure
///   `ic` is the number of input channels;
///   `c` * `expansion` is the number of output channels
struct BasicBlock: torch::nn::Module {
    static constexpr int expansion = 1;

    int64_t s;
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d bn1, bn2;
    torch::nn::Sequential shortcut;

    BasicBlock(int64_t ic, int64_t c, int64_t s=1);

    torch::Tensor forward(torch::Tensor x);
};

/// Bottleneck block of ResNet structure
///   `ic` is the number of input channels;
///   `c` * `expansion` is the number of output channels
struct Bottleneck: torch::nn::Module {
    static constexpr int expansion = 4;

    int64_t s;
    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::BatchNorm2d bn1, bn2, bn3;
    torch::nn::Sequential shortcut;

    Bottleneck(int64_t ic, int64_t c, int64_t s=1);

    torch::Tensor forward(torch::Tensor x);
};

template <typename Block>
struct ResNet: torch::nn::Module {
    uint64_t current_c;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    torch::nn::Linear linear;

    ResNet(int64_t b1, int64_t b2, int64_t b3, int64_t b4,
           int64_t num_classes=10);

    torch::nn::Sequential make_layer(int64_t c, int64_t n, int64_t s=1);

    torch::Tensor forward(const torch::Tensor& x);
};

static ResNet<BasicBlock> ResNet18() { return {2, 2, 2, 2}; }

static ResNet<BasicBlock> ResNet34() { return {3, 4, 6, 3}; }

static ResNet<Bottleneck> ResNet50() { return {3, 4, 6, 3}; }

static ResNet<Bottleneck> ResNet101() { return {3, 4, 23, 2}; }

static ResNet<Bottleneck> ResNet152() { return {3, 8, 36, 3}; }

} // End namespace rnlt
