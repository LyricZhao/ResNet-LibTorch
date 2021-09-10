#pragma once

#include <string>
#include <torch/torch.h>

namespace rnlt {

class CIFAR10: public torch::data::datasets::Dataset<CIFAR10> {
public:
    torch::Tensor images, labels;

    explicit CIFAR10(const std::string& path, bool train=true);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;
};

} // End namespace rnlt
