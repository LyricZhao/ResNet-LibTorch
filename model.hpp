#pragma once

#include <torch/torch.h>

torch::nn::Conv2dOptions conv_options(int64_t ic, int64_t oc, int64_t k,
                                      int64_t s, int64_t p= 0, bool bias= false) {
    torch::nn::Conv2dOptions options(ic, oc, k);
    options.stride() = s;
    options.padding() = p;
    options.bias() = bias;
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

    BasicBlock(int64_t ic, int64_t c, int64_t s=1):
            conv1(conv_options(ic, c, 3, s, 1)),
            bn1(c),
            conv2(conv_options(c, c * expansion, 3, 1, 1)),
            bn2(c * expansion),
            s(s) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        if (s != 1 or ic != c * expansion) {
            shortcut = torch::nn::Sequential(
                torch::nn::Conv2d(conv_options(ic, c * expansion, 1, s)),
                torch::nn::BatchNorm2d(c * expansion)
            );
            register_module("shortcut", shortcut);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1(conv1(x)));
        out = bn2(conv2(out));
        // TODO: May check whether empty shortcut works
        out += shortcut->forward(x);
        out = torch::relu(out);
        return out;
    }
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

    Bottleneck(int64_t ic, int64_t c, int64_t s=1):
            conv1(conv_options(ic, c, 1, s)),
            bn1(c),
            conv2(conv_options(c, c, 3, 1, 1)),
            bn2(c),
            conv3(conv_options(c, c * expansion, 1, 1)),
            bn3(c * expansion),
            s(s) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("conv3", conv3);
        register_module("bn3", bn3);
        if (s != 1 or ic != c * expansion) {
            shortcut = torch::nn::Sequential(
                    torch::nn::Conv2d(conv_options(ic, c * expansion, 1, s)),
                    torch::nn::BatchNorm2d(c * expansion)
            );
            register_module("shortcut", shortcut);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto out = torch::relu(bn1(conv1(x)));
        out = bn2(conv2(out));
        out = bn3(conv3(out));
        // TODO: May check whether empty shortcut works
        out += shortcut->forward(x);
        out = torch::relu(out);
        return out;
    }
};

template<typename Block>
struct ResNet: torch::nn::Module {
    uint64_t current_c;
    torch::nn::Conv2d conv1;
    torch::nn::BatchNorm2d bn1;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    torch::nn::Linear linear;

    ResNet(int64_t b1, int64_t b2, int64_t b3, int64_t b4,
           int64_t num_classes=10):
            conv1(conv_options(3, 64, 3, 1, 1)),
            bn1(64),
            current_c(64),
            layer1(make_layer(64, b1, 1)),
            layer2(make_layer(128, b2, 2)),
            layer3(make_layer(256, b3, 2)),
            layer4(make_layer(512, b4, 2)),
            linear(512 * Block::expansion, num_classes) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("linear", linear);
        // TODO: Better initialize all the weights
    }

    torch::nn::Sequential make_layer(int64_t c, int64_t n, int64_t s= 1) {
        torch::nn::Sequential seq;
        for (int i = 0; i < n; ++ i, s = 1) {
            seq->push_back(Block(current_c, c, s));
            current_c = c * Block::expansion;
        }
        return seq;
    }

    torch::Tensor forward(const torch::Tensor& x) {
        auto out = torch::relu(bn1(conv1(x)));
        out = layer1->forward(out);
        out = layer2->forward(out);
        out = layer3->forward(out);
        out = layer4->forward(out);
        out = torch::avg_pool2d(out, 4);
        out = out.view({out.size(0), -1});
        out = linear(out);
        return out;
    }
};

static ResNet<BasicBlock> ResNet18(){ return ResNet<BasicBlock>(2, 2, 2, 2); }

static ResNet<BasicBlock> ResNet34(){ return ResNet<BasicBlock>(3, 4, 6, 3); }

static ResNet<Bottleneck> ResNet50(){ return ResNet<Bottleneck>(3, 4, 6, 3); }

static ResNet<Bottleneck> ResNet101(){ return ResNet<Bottleneck>(3, 4, 23, 2); }

static ResNet<Bottleneck> ResNet152(){ return ResNet<Bottleneck>(3, 8, 36, 3); }
