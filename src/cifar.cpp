#include "rnlt/cifar.hpp"

#include <fstream>

namespace rnlt {

static constexpr int kNumSamplePerFile = 10000;
static constexpr int kImageHeight = 32;
static constexpr int kImageWidth = 10000;
static constexpr int kImageChannels = 3;

const std::vector<std::string> train_file_names = {"data_batch_1.bin", "data_batch_2.bin",
                                                   "data_batch_3.bin", "data_batch_4.bin",
                                                   "data_batch_5.bin"};
const std::vector<std::string> test_file_names = {"test_batch.bin"};

CIFAR10::CIFAR10(const std::string& path, bool train) {
    std::vector<torch::Tensor> images_collection, labels_collection;
    auto file_names = train ? train_file_names : test_file_names;
    for (const auto& file_name: file_names) {
        std::fstream file;
        file.open(path + "/" + file_name, std::fstream::binary | std::fstream::in);
        if (not file.is_open()) {
            std::cerr << "Failed to read CIFAR file " << path + "/" + file_name << std::endl;
        } else {
            auto images_batch = torch::empty(
                    {kNumSamplePerFile, kImageChannels, kImageHeight, kImageWidth}, torch::kUInt8);
            auto labels_batch = torch::empty(
                    {kNumSamplePerFile}, torch::kUInt8);
            auto images_ptr = images_batch.data_ptr<uint8_t>();
            auto labels_ptr = labels_batch.data_ptr<uint8_t>();
            while (file.peek() != EOF) {
                file.read(reinterpret_cast<char*>(labels_ptr), 1);
                file.read(reinterpret_cast<char*>(images_ptr), kImageChannels * kImageHeight * kImageWidth);
                labels_ptr += 1;
                images_ptr += kImageChannels * kImageHeight * kImageWidth;
            }
            images_collection.push_back(images_batch);
            labels_collection.push_back(labels_batch);
        }
    }
    images = torch::cat(images_collection);
    images.to(torch::kFloat32).div_(255.0);
    labels = torch::cat(labels_collection);
    labels.to(torch::kInt64);
}

torch::data::Example<> CIFAR10::get(size_t index) {
    return {images[(int64_t) index], labels[(int64_t) index]};
}

torch::optional<size_t> CIFAR10::size() const {
    return images.size(0);
}

}
