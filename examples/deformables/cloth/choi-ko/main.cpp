#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.choi_ko;

int main() {
    try {
        physica::examples::cloth::choi_ko::Simulation simulation{};
        const physica::examples::cloth::choi_ko::Summary summary = simulation.run();

        std::println("algorithm=ChoiKo2002-2003ImmediateBuckling");
        std::println("integration=second-order-semi-implicit-BDF2 equation=Choi-Ko-Eq20 solver=GPU-block-PCG preconditioner=exact-3x3-block-Jacobi");
        std::println("membrane=tension-only-UV-and-diagonal bending=opposite-vertex-inverse-sinc imperfection=Eq11 jacobian=Eq12-radial-only");
        std::println("damping=material-intrinsic membrane_Kx=retained-symmetric-Cdot-H-term omitted=non-symmetric-gradC-outer-Hv type2=relative-velocity");
        std::println("mesh={}x{} width={:.6f} height={:.6f} anchors=top-corners fixed_particles=({}, {}) gravity=(0,{:.6f},0) mass={:.6f}", simulation.rows, simulation.columns, simulation.width, simulation.height, simulation.fixed_particles[0], simulation.fixed_particles[1], simulation.gravity_y, simulation.mass);
        std::println("stiffness_uv=({:.3e},{:.3e}) stiffness_diagonal=({:.3e},{:.3e}) stiffness_bend=({:.6f},{:.6f}) imperfection={:.6f}", simulation.stretch_u_stiffness, simulation.stretch_v_stiffness, simulation.diagonal_u_stiffness, simulation.diagonal_v_stiffness, simulation.bend_u_stiffness, simulation.bend_v_stiffness, simulation.imperfection_stiffness);
        std::println("damping_uv=({:.6f},{:.6f}) damping_diagonal=({:.6f},{:.6f}) damping_bending={:.6f}", simulation.stretch_u_damping, simulation.stretch_v_damping, simulation.diagonal_u_damping, simulation.diagonal_v_damping, simulation.bending_damping);
        std::println("initial_in_plane_scale={:.6f} initial_perturbation_z={:.6f} initial_velocity_scale={:.6f} BDF2_history=explicit-x_n-v_n-x_n_minus_1-v_n_minus_1", simulation.initial_in_plane_scale, simulation.initial_perturbation, simulation.initial_velocity_scale);
        std::println("frames={} dt={:.9f} physical_time={:.6f} pcg_iterations={}", summary.frames, simulation.time_step, summary.physical_time, simulation.pcg_iteration_count);
        std::println("maximum_tension_conditions=(u={:.9e},v={:.9e},diagonal_u={:.9e},diagonal_v={:.9e}) active_condition_count={}", summary.maximum_tension_conditions[0], summary.maximum_tension_conditions[1], summary.maximum_tension_conditions[2], summary.maximum_tension_conditions[3], summary.active_tension_condition_count);
        std::println("buckled_hinge_count={} maximum_hinge_curvature={:.9e}", summary.buckled_hinge_count, summary.maximum_hinge_curvature);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("maximum_position_magnitude={:.9f} maximum_velocity_magnitude={:.9f}", summary.maximum_position_magnitude, summary.maximum_velocity_magnitude);
        std::println("probe_particle={} probe_displacement={:.9e} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_displacement, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Choi-Ko example failed: {}", error.what());
        return 1;
    }
}
