#include "rnlt/model.hpp"

#include <iostream>
#include <torch/torch.h>

int main() {
    torch::Device device("cpu");
    if (torch::cuda::is_available())
        device = torch::Device("cuda:0");

    auto t = torch::rand({4, 3, 224, 224}).to(device);
    auto net = rnlt::ResNet50();
    net.to(device);

    t = net.forward(t);
    std::cout << t.sizes() << std::endl;

    return 0;
}
