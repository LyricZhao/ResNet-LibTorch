#include "rnlt/cifar.hpp"
#include "rnlt/model.hpp"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

int main(int argc, char** argv) {
    // Parse arguments
    po::options_description desc("Train CIFAR-10 with ResNet");
    desc.add_options()
        ("path", po::value<std::string>()->required(), "CIFAR-10 path");
    po::variables_map args;
    po::store(po::parse_command_line(argc, argv, desc), args);
    po::notify(args);

    // Read dataset
    rnlt::CIFAR10 dataset(args["path"].as<std::string>());
    return 0;
}
