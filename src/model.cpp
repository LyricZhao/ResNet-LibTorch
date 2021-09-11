#include "rnlt/model.hpp"

namespace rnlt {

BasicBlock::BasicBlock(int64_t ic, int64_t c, int64_t s):
        conv1(conv_options(ic, c, 3, s, 1)),
        bn1(c),
        conv2(conv_options(c, c * expansion, 3, 1, 1)),
        bn2(c * expansion),
        s(s) {
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    if (s != 1 or ic != c * expansion)
        shortcut = torch::nn::Sequential(
                torch::nn::Conv2d(conv_options(ic, c * expansion, 1, s)),
                torch::nn::BatchNorm2d(c * expansion)
        );
    else
        shortcut = torch::nn::Sequential(torch::nn::Identity());
    register_module("shortcut", shortcut);
}

torch::Tensor BasicBlock::forward(const torch::Tensor& x) {
    auto out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    out += shortcut->forward(x);
    out = torch::relu(out);
    return out;
}

Bottleneck::Bottleneck(int64_t ic, int64_t c, int64_t s):
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
    if (s != 1 or ic != c * expansion)
        shortcut = torch::nn::Sequential(
                torch::nn::Conv2d(conv_options(ic, c * expansion, 1, s)),
                torch::nn::BatchNorm2d(c * expansion)
        );
    else
        shortcut = torch::nn::Sequential(torch::nn::Identity());
    register_module("shortcut", shortcut);
}

torch::Tensor Bottleneck::forward(const torch::Tensor& x) {
    auto out = torch::relu(bn1(conv1(x)));
    out = bn2(conv2(out));
    out = bn3(conv3(out));
    out += shortcut->forward(x);
    out = torch::relu(out);
    return out;
}

template <typename Block>
ResNet<Block>::ResNet(int64_t b1, int64_t b2, int64_t b3, int64_t b4, int64_t num_classes):
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

    // Initialize
    for (auto &module: modules(false)) {
        if (module->name() == "torch::nn::Conv2dImpl") {
            for (auto &p: module->named_parameters(false))
                torch::nn::init::xavier_normal_(p.value());
        } else if (module->name() == "torch::nn::BatchNorm2dImpl") {
            for (auto &p: module->named_parameters(false))
                torch::nn::init::constant_(p.value(), p.key() == "weight");
        }
    }
}

template <typename Block>
torch::nn::Sequential ResNet<Block>::make_layer(int64_t c, int64_t n, int64_t s) {
    torch::nn::Sequential seq;
    for (int i = 0; i < n; ++ i, s = 1) {
        seq->push_back(Block(current_c, c, s));
        current_c = c * Block::expansion;
    }
    return seq;
}

template <typename Block>
torch::Tensor ResNet<Block>::forward(const torch::Tensor& x) {
    auto out = torch::relu(bn1(conv1(x)));
    // Reduce the feature maps by 32x
    out = layer1->forward(out);
    out = layer2->forward(out);
    out = layer3->forward(out);
    out = layer4->forward(out);
    out = torch::avg_pool2d(out, 4);
    out = out.view({out.size(0), -1});
    out = linear(out);
    return out;
}

// Template instantiation for further linking
template struct ResNet<BasicBlock>;
template struct ResNet<Bottleneck>;

} // End namespace rnlt
