module;

#include "kernels.h"
#include <physica/cuda.h>
#include <spectra/sdk/cuda_types.h>

module physica.example.fluids.liquid.flip_apic_dam_break.module;

import std;
import physica.example.fluids.liquid.flip_apic_dam_break;
import spectra.sdk;
import spectra.sdk.cuda;

namespace physica::examples::flip_apic_dam_break {
    Module::Module(const Settings source, const std::filesystem::path&, const spectra::sdk::SceneInputs&) : settings(source), simulation(std::make_unique<Simulation>()) {}

    Module::~Module() = default;

    void Module::setup(spectra::sdk::cuda::Setup& setup) {
        setup.particles<"flip-particles">(Simulation::maximum_particle_count, Simulation::particle_radius);
        setup.particles<"apic-particles">(Simulation::maximum_particle_count, Simulation::particle_radius);
        setup.volume<"flip-grid">({Simulation::resolution[0], Simulation::resolution[1], Simulation::resolution[2]});
        setup.volume<"apic-grid">({Simulation::resolution[0], Simulation::resolution[1], Simulation::resolution[2]});
    }

    void Module::reset(const std::uint64_t) {
        simulation->reset();
    }

    void Module::step(const double seconds) {
        simulation->step(seconds);
    }

    void Module::publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame) {
        Simulation& state                                  = *simulation;
        spectra::sdk::cuda::Frame frame                    = output.begin(state.stream.get());
        const spectra::sdk::cuda::Particles flip_particles = frame.particles<"flip-particles">(state.flip_state.particle_count);
        spectra_cuda::write_flip_particles(state.stream, state.flip_state.particle_count, 0.0F, state.flip_state.positions.x.data(), state.flip_state.positions.y.data(), state.flip_state.positions.z.data(), state.flip_state.velocities.x.data(), state.flip_state.velocities.y.data(), state.flip_state.velocities.z.data(), flip_particles.positions.data(), flip_particles.field<"velocity", spectra::sdk::Float3>().data(), flip_particles.field<"speed", float>().data());
        const spectra::sdk::cuda::Particles apic_particles = frame.particles<"apic-particles">(state.apic_state.particle_count);
        const auto& affine                                 = state.apic_state.transfer.affine;
        spectra_cuda::write_apic_particles(state.stream, state.apic_state.particle_count, 0.95F, state.apic_state.positions.x.data(), state.apic_state.positions.y.data(), state.apic_state.positions.z.data(), state.apic_state.velocities.x.data(), state.apic_state.velocities.y.data(), state.apic_state.velocities.z.data(), affine.c00.data(), affine.c01.data(), affine.c02.data(), affine.c10.data(), affine.c11.data(), affine.c12.data(), affine.c20.data(), affine.c21.data(), affine.c22.data(), apic_particles.positions.data(), apic_particles.field<"velocity", spectra::sdk::Float3>().data(), apic_particles.field<"speed", float>().data(), apic_particles.field<"affine-magnitude", float>().data());
        const spectra::sdk::cuda::Volume flip_grid       = frame.volume<"flip-grid">();
        const spectra::sdk::cuda::MacField flip_velocity = flip_grid.field<"velocity">();
        ::cuda::copy_bytes(state.stream, state.flip_cache.grid.velocity.x, ::cuda::std::span{flip_velocity.x.data(), flip_velocity.x.size()});
        ::cuda::copy_bytes(state.stream, state.flip_cache.grid.velocity.y, ::cuda::std::span{flip_velocity.y.data(), flip_velocity.y.size()});
        ::cuda::copy_bytes(state.stream, state.flip_cache.grid.velocity.z, ::cuda::std::span{flip_velocity.z.data(), flip_velocity.z.size()});
        const std::span<float> flip_pressure          = flip_grid.field<"pressure", float>();
        const std::span<float> flip_divergence        = flip_grid.field<"divergence", float>();
        const std::span<float> flip_level_set         = flip_grid.field<"level-set", float>();
        const std::span<std::uint32_t> flip_cell_type = flip_grid.field<"cell-type", std::uint32_t>();
        ::cuda::copy_bytes(state.stream, state.flip_cache.grid.pressure.values, ::cuda::std::span{flip_pressure.data(), flip_pressure.size()});
        ::cuda::copy_bytes(state.stream, state.flip_cache.grid.divergence.values, ::cuda::std::span{flip_divergence.data(), flip_divergence.size()});
        ::cuda::copy_bytes(state.stream, state.flip_cache.grid.level_set.values, ::cuda::std::span{flip_level_set.data(), flip_level_set.size()});
        ::cuda::copy_bytes(state.stream, state.flip_cache.grid.cell_types.values, ::cuda::std::span{flip_cell_type.data(), flip_cell_type.size()});

        const spectra::sdk::cuda::Volume apic_grid       = frame.volume<"apic-grid">();
        const spectra::sdk::cuda::MacField apic_velocity = apic_grid.field<"velocity">();
        ::cuda::copy_bytes(state.stream, state.apic_cache.grid.velocity.x, ::cuda::std::span{apic_velocity.x.data(), apic_velocity.x.size()});
        ::cuda::copy_bytes(state.stream, state.apic_cache.grid.velocity.y, ::cuda::std::span{apic_velocity.y.data(), apic_velocity.y.size()});
        ::cuda::copy_bytes(state.stream, state.apic_cache.grid.velocity.z, ::cuda::std::span{apic_velocity.z.data(), apic_velocity.z.size()});
        const std::span<float> apic_pressure          = apic_grid.field<"pressure", float>();
        const std::span<float> apic_divergence        = apic_grid.field<"divergence", float>();
        const std::span<float> apic_level_set         = apic_grid.field<"level-set", float>();
        const std::span<std::uint32_t> apic_cell_type = apic_grid.field<"cell-type", std::uint32_t>();
        ::cuda::copy_bytes(state.stream, state.apic_cache.grid.pressure.values, ::cuda::std::span{apic_pressure.data(), apic_pressure.size()});
        ::cuda::copy_bytes(state.stream, state.apic_cache.grid.divergence.values, ::cuda::std::span{apic_divergence.data(), apic_divergence.size()});
        ::cuda::copy_bytes(state.stream, state.apic_cache.grid.level_set.values, ::cuda::std::span{apic_level_set.data(), apic_level_set.size()});
        ::cuda::copy_bytes(state.stream, state.apic_cache.grid.cell_types.values, ::cuda::std::span{apic_cell_type.data(), apic_cell_type.size()});

        frame.metric<"step">().upload(state.step_index);
        frame.metric<"time">().upload(state.physical_time);
        frame.metric<"flip-particle-count">().upload(static_cast<std::uint64_t>(state.flip_state.particle_count));
        frame.metric<"apic-particle-count">().upload(static_cast<std::uint64_t>(state.apic_state.particle_count));
        frame.metric<"flip-substeps">().upload(state.flip_substep_count);
        frame.metric<"apic-substeps">().upload(state.apic_substep_count);
        frame.metric<"flip-pressure-iterations">().upload(static_cast<std::uint64_t>(state.flip_cache.diagnostics.projection.iterations));
        frame.metric<"apic-pressure-iterations">().upload(static_cast<std::uint64_t>(state.apic_cache.diagnostics.projection.iterations));
        frame.metric<"flip-pressure-residual">().upload(state.flip_cache.diagnostics.projection.relative_residual);
        frame.metric<"apic-pressure-residual">().upload(state.apic_cache.diagnostics.projection.relative_residual);
        frame.metric<"flip-divergence-before">().upload(state.flip_cache.diagnostics.divergence_before_projection.l2);
        frame.metric<"flip-divergence-after">().upload(state.flip_cache.diagnostics.divergence_after_projection.l2);
        frame.metric<"apic-divergence-before">().upload(state.apic_cache.diagnostics.divergence_before_projection.l2);
        frame.metric<"apic-divergence-after">().upload(state.apic_cache.diagnostics.divergence_after_projection.l2);
        frame.metric<"flip-kinetic-energy">().upload(state.flip_particle_diagnostics.kinetic_energy);
        frame.metric<"apic-kinetic-energy">().upload(state.apic_particle_diagnostics.kinetic_energy);
        frame.metric<"flip-linear-momentum">().upload(spectra::sdk::Float3{state.flip_particle_diagnostics.linear_momentum.x, state.flip_particle_diagnostics.linear_momentum.y, state.flip_particle_diagnostics.linear_momentum.z});
        frame.metric<"apic-linear-momentum">().upload(spectra::sdk::Float3{state.apic_particle_diagnostics.linear_momentum.x, state.apic_particle_diagnostics.linear_momentum.y, state.apic_particle_diagnostics.linear_momentum.z});
        frame.metric<"flip-angular-momentum">().upload(spectra::sdk::Float3{state.flip_particle_diagnostics.angular_momentum.x, state.flip_particle_diagnostics.angular_momentum.y, state.flip_particle_diagnostics.angular_momentum.z});
        frame.metric<"apic-angular-momentum">().upload(spectra::sdk::Float3{state.apic_particle_diagnostics.angular_momentum.x, state.apic_particle_diagnostics.angular_momentum.y, state.apic_particle_diagnostics.angular_momentum.z});
        frame.commit();
    }
} // namespace physica::examples::flip_apic_dam_break
