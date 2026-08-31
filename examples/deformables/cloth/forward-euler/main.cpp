#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.forward_euler;

int main() {
    try {
        physica::examples::cloth::forward_euler::Simulation simulation{};
        const physica::examples::cloth::forward_euler::Summary summary = simulation.run();
        std::println("algorithm=ForwardEuler");
        std::println("update=x[n+1]=x[n]+dt*v[n]; v[n+1]=v[n]+dt*f[n]/m");
        std::println("mesh={}x{} anchors=top-corners gravity_y={:.3f} wind=off", simulation.rows, simulation.columns, simulation.gravity_y);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("probe_y initial={:.9f} first={:.9f} final={:.9f}", summary.initial_probe_y, summary.first_frame_probe_y, summary.final_probe_position.y);
        std::println("probe_position=({:.9f},{:.9f},{:.9f})", summary.final_probe_position.x, summary.final_probe_position.y, summary.final_probe_position.z);
        std::println("probe_velocity=({:.9f},{:.9f},{:.9f})", summary.final_probe_velocity.x, summary.final_probe_velocity.y, summary.final_probe_velocity.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth Forward Euler example failed: {}", error.what());
        return 1;
    }
}
