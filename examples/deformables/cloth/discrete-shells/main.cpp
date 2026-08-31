#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.discrete_shells;

int main() {
    try {
        physica::examples::cloth::discrete_shells::Simulation simulation{};
        const physica::examples::cloth::discrete_shells::Summary summary = simulation.run();

        std::println("algorithm=GrinspunHiraniDesbrunSchroder2003DiscreteShells");
        std::println("integration=Newmark beta=1/4 gamma=1/2 Newton=exact-energy-Hessian globalization=GPU-batched-Armijo solver=GPU-block-PCG");
        std::println("membrane=kL*sum(l0*(1-l/l0)^2)+kA*sum(A0*(1-A/A0)^2) half_factor=none");
        std::println("bending=kB*sum(3*l0^2/(A01+A02)*(theta-theta0)^2) rest_dihedral=signed-from-curved-3D-rest");
        std::println("damping=paper-forward-angle-difference force=-c*w*((theta_next-theta_current)/dt)*grad(theta_next) exact-symmetric-Jacobian");
        std::println("mesh={}x{} beam_length={:.6f} beam_width={:.6f} crease_depth={:.6f} fixed_wall_vertices={}", simulation.rows, simulation.columns, simulation.beam_length, simulation.beam_width, simulation.crease_depth, simulation.fixed_particles.size());
        std::println("surface_density={:.6f} mass=lumped-one-third-rest-triangle-area gravity=(0,0,{:.6f})", simulation.surface_density, simulation.gravity_z);
        std::println("stiffness_length={:.3e} stiffness_area={:.3e} stiffness_bending={:.6f} damping_bending={:.6f}", simulation.length_stiffness, simulation.area_stiffness, simulation.bending_stiffness, simulation.bending_damping);
        std::println("newton_iterations={} pcg_iterations={} line_search_candidates={} initial_tip_velocity={:.6f}", simulation.newton_iteration_count, simulation.pcg_iteration_count, simulation.line_search_candidate_count, simulation.initial_tip_velocity);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("curved_rest_hinge_count={} maximum_absolute_rest_dihedral={:.9e}", summary.curved_rest_hinge_count, summary.maximum_absolute_rest_dihedral);
        std::println("maximum_relative_edge_length_error={:.9e} maximum_relative_triangle_area_error={:.9e}", summary.maximum_relative_edge_length_error, summary.maximum_relative_triangle_area_error);
        std::println("maximum_absolute_dihedral_delta={:.9e}", summary.maximum_absolute_dihedral_delta);
        std::println("total_membrane_energy={:.9e} total_bending_energy={:.9e} total_damping_potential={:.9e}", summary.total_membrane_energy, summary.total_bending_energy, summary.total_damping_potential);
        std::println("final_regularization_shift={:.9e} final_accepted_step={:.9f} final_candidate={}", summary.regularization_shift, summary.accepted_step_size, summary.accepted_line_search_candidate);
        std::println("maximum_fixed_position_error={:.9e} maximum_position_magnitude={:.9f} maximum_velocity_magnitude={:.9f}", summary.maximum_fixed_position_error, summary.maximum_position_magnitude, summary.maximum_velocity_magnitude);
        std::println("probe_particle={} probe_displacement={:.9e} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_displacement, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Discrete Shells example failed: {}", error.what());
        return 1;
    }
}
