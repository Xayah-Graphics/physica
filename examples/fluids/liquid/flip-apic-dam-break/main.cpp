#include <physica/cuda.h>

import std;
import physica.example.fluids.liquid.flip_apic_dam_break;

int main() {
    try {
        physica::examples::flip_apic_dam_break::Simulation simulation{};
        std::println("FLIP/APIC dam break with {} initial particles per solver", physica::examples::flip_apic_dam_break::Simulation::initial_particle_count);
        for (std::uint32_t frame = 0u; frame < 120u; ++frame) {
            simulation.step(1.0 / 120.0);
            if (simulation.step_index % 30u == 0u)
                std::println("frame {:3} time {:.3f} s | FLIP particles {} div {:.3e} PCG {} {:.3e} | APIC particles {} div {:.3e} PCG {} {:.3e}", simulation.step_index, simulation.physical_time,
                    simulation.flip_state.particle_count, simulation.flip_cache.diagnostics.divergence_after_projection.l2, simulation.flip_cache.diagnostics.projection.iterations, simulation.flip_cache.diagnostics.projection.relative_residual,
                    simulation.apic_state.particle_count, simulation.apic_cache.diagnostics.divergence_after_projection.l2, simulation.apic_cache.diagnostics.projection.iterations, simulation.apic_cache.diagnostics.projection.relative_residual);
        }
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "FLIP/APIC dam-break example failed: {}", error.what());
        return 1;
    }
}
