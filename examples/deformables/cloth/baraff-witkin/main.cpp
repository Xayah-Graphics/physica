#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.baraff_witkin;

int main() {
    try {
        physica::examples::cloth::baraff_witkin::Simulation simulation{};
        const physica::examples::cloth::baraff_witkin::Summary summary = simulation.run();

        std::println("algorithm=BaraffWitkin1998");
        std::println("integration=linearized-backward-Euler system=A=M-h*dfdv-h^2*dfdx solver=GPU-block-PCG preconditioner=exact-3x3-block-Jacobi");
        std::println("membrane=UV-area-scaled-stretch-shear bending=signed-dihedral-rest-angle anisotropy=warp-weft damping=condition-matching");
        std::println("pcg_iterations={} stretch_stiffness=({:.3e},{:.3e}) shear_stiffness={:.3e} bend_stiffness=({:.6f},{:.6f})", simulation.pcg_iteration_count, simulation.stretch_u_stiffness, simulation.stretch_v_stiffness, simulation.shear_stiffness, simulation.bend_u_stiffness, simulation.bend_v_stiffness);
        std::println("mesh={}x{} width={:.6f} height={:.6f} anchors=top-corners fixed_particles=({}, {}) gravity=(0,{:.6f},0) mass={:.6f}", simulation.rows, simulation.columns, simulation.width, simulation.height, simulation.fixed_particles[0], simulation.fixed_particles[1], simulation.gravity_y, simulation.mass);
        std::println("initial_perturbation_z={:.6f} profile=sin(row)*sin(2*column)", simulation.initial_perturbation);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("maximum_material_stretch={:.9f}", summary.maximum_material_stretch);
        std::println("maximum_absolute_material_shear={:.9e}", summary.maximum_absolute_material_shear);
        std::println("maximum_fixed_position_error={:.9e}", summary.maximum_fixed_position_error);
        std::println("probe_particle={} probe_position=({:.9f},{:.9f},{:.9f})", simulation.probe_particle, summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Baraff-Witkin example failed: {}", error.what());
        return 1;
    }
}
