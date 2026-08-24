#include <physica/cuda.h>

import std;
import physica.example.fluids.gas.keyframe_smoke;

int main(const int argument_count, char** arguments) {
    const std::string_view command = arguments[1];
    if (command == "verify") {
        physica::examples::keyframe_smoke::Experiment experiment{'S'};
        experiment.verify(arguments[2]);
        return 0;
    }
    if (command == "optimize") {
        physica::examples::keyframe_smoke::Experiment experiment{arguments[2][0]};
        experiment.optimize(arguments[3]);
        return 0;
    }
    if (command == "compose") {
        physica::examples::keyframe_smoke::compose(arguments[2]);
        return 0;
    }
    std::unreachable();
}
