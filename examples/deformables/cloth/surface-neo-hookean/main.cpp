#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.surface_neo_hookean;

int main() {
    try {
        physica::examples::cloth::surface_neo_hookean::Simulation simulation{};
        const physica::examples::cloth::surface_neo_hookean::Summary summary = simulation.run();

        std::println("algorithm=IntrinsicSurfaceNeoHookeanCST");
        std::println("integration=backward-Euler incremental-potential Newton=exact-gradient-and-Hessian globalization=GPU-Weyl-domain-bound-plus-batched-Armijo solver=GPU-block-PCG");
        std::println("material=intrinsic-2D-surface-Neo-Hookean plane-stress-small-strain-calibration energy=thickness*rest-UV-area*(mu/2*(tr(F^T*F)-2)-mu*log(J)+lambda/2*log(J)^2)");
        std::println("regularization=free-free-Gershgorin-shift raw-Hessian=preserved preconditioner=exact-3x3-block-Jacobi");
        std::println("mesh={}x{} width={:.6f} height={:.6f} anchors=top-corners fixed_particles=({}, {}) gravity=(0,{:.6f},0) mass={:.6f}", simulation.rows, simulation.columns, simulation.width, simulation.height, simulation.fixed_particles[0], simulation.fixed_particles[1], simulation.gravity_y, simulation.mass);
        std::println("young_modulus={:.3e} poisson_ratio={:.6f} thickness={:.7f}", simulation.young_modulus, simulation.poisson_ratio, simulation.thickness);
        std::println("newton_iterations={} pcg_iterations={} line_search_candidates={} contraction=0.5 domain_safety={:.3f}", simulation.newton_iteration_count, simulation.pcg_iteration_count, simulation.line_search_candidate_count, simulation.domain_safety);
        std::println("initial_in_plane_compression={:.6f} initial_perturbation_z={:.6f} initial_velocity_z={:.6f}", simulation.initial_compression, simulation.initial_perturbation, simulation.initial_velocity);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("surface_jacobian_min={:.9e} surface_jacobian_max={:.9e} maximum_absolute_log_surface_jacobian={:.9e}", summary.minimum_surface_jacobian, summary.maximum_surface_jacobian, summary.maximum_absolute_log_surface_jacobian);
        std::println("total_elastic_energy={:.9e}", summary.total_elastic_energy);
        std::println("final_regularization_shift={:.9e} final_domain_step={:.9f} final_accepted_step={:.9f} final_candidate={}", summary.regularization_shift, summary.maximum_domain_step, summary.accepted_step_size, summary.accepted_line_search_candidate);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("maximum_position_magnitude={:.9f} maximum_velocity_magnitude={:.9f}", summary.maximum_position_magnitude, summary.maximum_velocity_magnitude);
        std::println("probe_particle={} probe_displacement={:.9e} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_displacement, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Surface Neo-Hookean FEM example failed: {}", error.what());
        return 1;
    }
}
