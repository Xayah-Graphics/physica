#include <physica/cuda.h>

import std;
import physica.example.fluids.gas.keyframe_smoke;

int main() {
    const std::filesystem::path results_directory = "output/keyframe-smoke-gpu-final";
    physica::examples::keyframe_smoke::Experiment experiment{'S'};
    experiment.optimize(results_directory / "final/S");
}
