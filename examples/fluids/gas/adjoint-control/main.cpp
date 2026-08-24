#include <physica/cuda.h>

import std;
import physica.example.fluids.gas.adjoint_control;

int main(const int argument_count, char** arguments) {
    const std::string_view command = arguments[1];
    if (command == "verify") {
        physica::examples::adjoint_control::run_verification(arguments[2], arguments[3]);
        return 0;
    }
    if (command == "benchmark") {
        physica::examples::adjoint_control::benchmark(arguments[2], arguments[3]);
        return 0;
    }
    if (command == "bunny") {
        physica::examples::adjoint_control::run_bunny(arguments[2], arguments[3]);
        return 0;
    }
    std::unreachable();
}
