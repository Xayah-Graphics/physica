#include <physica/cuda.h>

import std;
import physica.example.deformables.cloth.analytic_collision;

int main() {
    try {
        physica::examples::cloth::analytic_collision::Simulation simulation{};
        const physica::examples::cloth::analytic_collision::Summary summary = simulation.run();
        std::println("algorithm=AnalyticCollision");
        std::println("collision=vertex-discrete-frictionless plane+sphere thickness={:.6f}", simulation.thickness);
        std::println("mesh={}x{} anchors=none gravity_y={:.3f}", simulation.rows, simulation.columns, simulation.gravity_y);
        std::println("sphere center=({:.6f},{:.6f},{:.6f}) radius={:.6f} plane_y={:.6f}", simulation.sphere_center.x, simulation.sphere_center.y, simulation.sphere_center.z, simulation.sphere_radius, simulation.plane_offset);
        std::println("frames={} dt={:.9f} physical_time={:.6f}", summary.frames, simulation.time_step, summary.physical_time);
        std::println("minimum_clearance plane={:.9f} sphere={:.9f}", summary.minimum_plane_clearance, summary.minimum_sphere_clearance);
        std::println("contact_vertices plane={} sphere={}", summary.plane_contact_vertices, summary.sphere_contact_vertices);
        std::println("center_position=({:.9f},{:.9f},{:.9f})", summary.final_center_position.x, summary.final_center_position.y, summary.final_center_position.z);
        std::println("center_velocity=({:.9f},{:.9f},{:.9f})", summary.final_center_velocity.x, summary.final_center_velocity.y, summary.final_center_velocity.z);
        std::println("corner_position=({:.9f},{:.9f},{:.9f})", summary.final_corner_position.x, summary.final_corner_position.y, summary.final_corner_position.z);
        return 0;
    } catch (const std::exception& error) {
        std::println(std::cerr, "cloth analytic collision example failed: {}", error.what());
        return 1;
    }
}
