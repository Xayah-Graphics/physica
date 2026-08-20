#include <cuda/__functional/call_or.h>
#include <cuda/buffer>
#include <cuda/stream>
#include <cstdio>

import std;
import physica.example.fluids.liquid.pbf_dam_break;

int main() {
    try {
        physica::examples::pbf_dam_break::Simulation simulation{};
        std::println("PBF dam break with {} particles", physica::examples::pbf_dam_break::Simulation::particle_count);
        for (std::uint32_t step = 0u; step < 240u; ++step) {
            simulation.step();
            if (simulation.step_index % 60u == 0u) std::println("step {:3}  time {:.3f} s", simulation.step_index, simulation.physical_time);
        }
        return 0;
    } catch (const std::exception& error) {
        std::println(stderr, "PBF dam-break example failed: {}", error.what());
        return 1;
    }
}
