#pragma once

#include <torch/torch.h>

torch::nn::Conv2dOptions ConvOptions(int64_t ic, int64_t oc, int64_t k,
                                     int64_t s, int64_t p=0, bool bias=false) {
    torch::nn::Conv2dOptions options(ic, oc, k);
    options.stride() = s;
    options.padding() = p;
    options.bias() = bias;
    return options;
}

struct BasicBlock: torch::nn::Module {
    static constexpr int expansion = 1;

    int64_t s;
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d bn1, bn2;
    torch::nn::Sequential shortcut;

    BasicBlock(int64_t ic, int64_t c, int64_t s=1):
            conv1(ConvOptions(ic, c, 3, s, 1)),
            bn1(c),
            conv2(ConvOptions(c, c * expansion, 3, 1, 1)),
            bn2(c * expansion),
            s(s) {
        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        if (s != 1 or ic != c * expansion) {
            shortcut = torch::nn::Sequential(
                torch::nn::Conv2d(ConvOptions(ic, c * expansion, 1, s)),
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

struct Bottleneck: torch::nn::Module {
    static constexpr int expansion = 4;

    int64_t s;
    torch::nn::Conv2d conv1, conv2, conv3;
    torch::nn::BatchNorm2d bn1, bn2, bn3;
    torch::nn::Sequential shortcut;

    Bottleneck(int64_t ic, int64_t c, int64_t s=1):
            conv1(ConvOptions(ic, c, 1, s)),
            bn1(c),
            conv2(ConvOptions(c, c, 3, 1, 1)),
            bn2(c),
            conv3(ConvOptions(c, c * expansion, 1, 1)),
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
                    torch::nn::Conv2d(ConvOptions(ic, c * expansion, 1, s)),
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
