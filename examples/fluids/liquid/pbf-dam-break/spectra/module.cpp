module;

#include "kernels.h"
#include <physica/cuda.h>

module physica.example.fluids.liquid.pbf_dam_break.module;

import std;
import physica.example.fluids.liquid.pbf_dam_break;
import spectra.sdk;
import spectra.sdk.cuda;

namespace physica::examples::pbf_dam_break {
    Module::Module(const Settings source, const std::filesystem::path&, const spectra::sdk::SceneInputs&) : settings(source), simulation(std::make_unique<Simulation>()) {}

    Module::~Module() = default;

    void Module::setup(spectra::sdk::cuda::Setup& setup) {
        setup.particles<"particles">(Simulation::particle_count, Simulation::particle_radius);
    }

    void Module::reset(const std::uint64_t) {
        simulation->reset();
    }

    void Module::step(const double) {
        simulation->step();
    }

    void Module::publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame) {
        Simulation& state                                = *simulation;
        spectra::sdk::cuda::Frame frame                  = output.begin(state.stream.get());
        const spectra::sdk::cuda::Particles particles    = frame.particles<"particles">(Simulation::particle_count);
        const std::span<spectra::sdk::Float3> positions  = particles.positions;
        const std::span<spectra::sdk::Float3> velocities = particles.field<"velocity", spectra::sdk::Float3>();
        spectra_cuda::write_vectors(state.stream, Simulation::particle_count, state.current_state.positions.x.data(), state.current_state.positions.y.data(), state.current_state.positions.z.data(), state.current_state.velocities.x.data(), state.current_state.velocities.y.data(), state.current_state.velocities.z.data(), positions.data(), velocities.data());
        const std::span<float> vorticity = particles.field<"vorticity", float>();
        ::cuda::copy_bytes(state.stream, state.step_cache.vorticity_magnitudes.values, ::cuda::std::span{vorticity.data(), vorticity.size()});
        frame.metric<"step">().upload(state.step_index);
        frame.metric<"time">().upload(state.physical_time);
        frame.commit();
    }
} // namespace physica::examples::pbf_dam_break
