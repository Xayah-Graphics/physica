#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.strain_based_dynamics;

int main() {
    try {
        physica::examples::cloth::strain_based_dynamics::Simulation simulation{};
        const physica::examples::cloth::strain_based_dynamics::Summary summary = simulation.run();
        std::println("algorithm=StrainBasedDynamics");
        std::println("prediction=v*=v+dt*(gravity+external_force/m); x*=x+dt*v*");
        std::println("constraints=Cuu=dot(fu,fu)-1; Cvv=dot(fv,fv)-1; Cuv=dot(fu,fv)");
        std::println("projection=triangle-colored Gauss-Seidel");
        std::println("mesh={}x{} anchors=top-corners mass={:.5f} gravity_y={:.3f} wind=off", simulation.rows, simulation.columns, simulation.mass, simulation.gravity_y);
        std::println("iterations={} stiffness_uu={:.3f} stiffness_vv={:.3f} stiffness_uv={:.3f}", simulation.iteration_count, simulation.stretch_stiffness_u, simulation.stretch_stiffness_v, simulation.shear_stiffness);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("max_constraint_error uu={:.9e} vv={:.9e} uv={:.9e}", summary.maximum_uu_error, summary.maximum_vv_error, summary.maximum_uv_error);
        std::println("max_fixed_error={:.9e}", summary.maximum_fixed_error);
        std::println("probe_position=({:.9f},{:.9f},{:.9f})", summary.probe_position.x, summary.probe_position.y, summary.probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.probe_velocity.x, summary.probe_velocity.y, summary.probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Strain-Based Dynamics example failed: {}", error.what());
        return 1;
    }
}
