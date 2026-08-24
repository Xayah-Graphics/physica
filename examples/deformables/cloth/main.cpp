#include <physica/cuda.h>
#include <cstdio>

import std;
import physica.example.deformables.cloth;

int main() {
    try {
        physica::examples::cloth::Simulation simulation{};
        std::println("64 x 96 aerodynamic flag in wind");
        for (std::uint32_t step = 0u; step < 1920u; ++step) {
            simulation.step();
            if (simulation.step_index % 60u == 0u) std::println("step {:4}  time {:.3f} s", simulation.step_index, simulation.physical_time);
        }
        return 0;
    } catch (const std::exception& error) {
        std::println(stderr, "cloth example failed: {}", error.what());
        return 1;
    }
}
