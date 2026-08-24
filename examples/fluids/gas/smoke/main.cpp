#include <physica/cuda.h>

import std;
import physica.example.fluids.gas.smoke;

int main() {
    physica::examples::smoke::Simulation simulation;
    for (std::uint32_t step = 0u; step < 240u; ++step) simulation.step();
    simulation.stream.sync();
    std::println("Double-jet thermal smoke completed {} steps ({:.3f} s)", simulation.step_index, simulation.physical_time);
}
