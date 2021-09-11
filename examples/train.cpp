#include "rnlt/cifar.hpp"
#include "rnlt/model.hpp"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char** argv) {
    // Parse arguments
    po::options_description desc("Train CIFAR-10 with ResNet");
    desc.add_options()
        ("path", po::value<std::string>()->required(), "CIFAR-10 path")
        ("lr", po::value<double>()->default_value(0.001), "Learning rate")
        ("epochs", po::value<int>()->default_value(6), "Number of epochs")
        ("batch_size", po::value<int>()->default_value(1024), "Batch size")
        ("workers", po::value<int>()->default_value(4), "Number of workers to process data");
    po::variables_map args;
    po::store(po::parse_command_line(argc, argv, desc), args);
    po::notify(args);

    // Read dataset
    std::cout << "Reading dataset ... " << std::flush;
    auto train_dataset = rnlt::CIFAR10(args["path"].as<std::string>())
        .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                  {0.2023, 0.1994, 0.2010}))
        .map(torch::data::transforms::Stack<>());
    auto train_data_loader = torch::data::make_data_loader(
            std::move(train_dataset), torch::data::DataLoaderOptions()
            .batch_size(args["batch_size"].as<int>())
            .workers(args["workers"].as<int>()));
    auto test_dataset = rnlt::CIFAR10(args["path"].as<std::string>(), false)
            .map(torch::data::transforms::Normalize<>({0.4914, 0.4822, 0.4465},
                                                      {0.2023, 0.1994, 0.2010}))
            .map(torch::data::transforms::Stack<>());
    auto test_data_loader = torch::data::make_data_loader(
            std::move(test_dataset), torch::data::DataLoaderOptions()
                    .batch_size(args["batch_size"].as<int>())
                    .workers(args["workers"].as<int>())
                    .enforce_ordering(true));
    std::cout << "OK!" << std::endl;

    // Load model and device
    std::cout << "Loading model ... " << std::flush;
    auto model = rnlt::ResNet18();
    auto device = torch::Device("cpu");
    if (torch::cuda::is_available())
        device = torch::Device("cuda:0");
    model.to(device);
    std::cout << "OK!" << std::endl;

    // Train model
    std::cout << "Begin training ..." << std::endl;
    torch::optim::AdamW optimizer(model.parameters(),torch::optim::AdamWOptions(args["lr"].as<double>())
            .weight_decay(5e-4));
    torch::optim::StepLR scheduler(optimizer, 3, 0.1);
    torch::nn::CrossEntropyLoss criterion;
    for (int i = 1, steps = 1, n = args["epochs"].as<int>(); i <= n; ++ i) {
        // Train
        model.train();
        for (auto& batch: *train_data_loader) {
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);
            optimizer.zero_grad();
            auto outs = model.forward(images);
            auto loss = criterion(outs, labels);
            std::cout << "\r > Loss@[" << steps ++ << ", " << i << "/" << n << "]: " << loss.item<float>() << std::flush;
            loss.backward();
            optimizer.step();
        }
        scheduler.step();
        std::cout << std::endl;

        // Evaluate
        model.eval();
        torch::NoGradGuard no_grad;
        int64_t num_samples = 0, num_corrects = 0;
        for (auto& batch: *test_data_loader) {
            auto images = batch.data.to(device);
            auto labels = batch.target.to(device);
            auto outs = model.forward(images);
            auto predictions = std::get<1>(torch::max(outs, 1));
            num_corrects += torch::sum(predictions == labels).item<int64_t>();
            num_samples += images.size(0);
            std::cout << "\r > Accuracy@" << i << ": " << (double) num_corrects / (double) num_samples
                      << " (" << num_corrects << "/" << num_samples << ")" << std::flush;
        }
    }
    std::cout << std::endl;
    return 0;
}
