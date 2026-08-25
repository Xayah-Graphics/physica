#include <physica/cuda.h>

import std;
import physica.example.fluids.gas.keyframe_smoke;

int main() {
    constexpr std::array letters{'S', 'M', 'O', 'K', 'E'};
    const std::filesystem::path results_directory = "output/keyframe-smoke";
    for (const char letter : letters) {
        physica::examples::keyframe_smoke::Experiment experiment{letter};
        experiment.optimize(results_directory / "final" / std::string(1u, letter));
    }
    physica::examples::keyframe_smoke::compose(results_directory);
}
