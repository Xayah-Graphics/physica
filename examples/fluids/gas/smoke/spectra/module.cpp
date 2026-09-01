module;

#include <physica/cuda.h>

module physica.example.fluids.gas.smoke.module;

import std;
import physica.example.fluids.gas.smoke;
import spectra.sdk;
import spectra.sdk.cuda;

namespace physica::examples::smoke {
    Module::Module(const Settings source, const std::filesystem::path&, const spectra::sdk::SceneInputs&) : settings(source), simulation(std::make_unique<Simulation>()) {}

    Module::~Module() = default;

    void Module::setup(spectra::sdk::cuda::Setup& setup) {
        setup.volume<"smoke">({Simulation::resolution[0], Simulation::resolution[1], Simulation::resolution[2]});
    }

    void Module::reset(const std::uint64_t) {
        simulation->reset();
    }

    void Module::step(const double) {
        simulation->step();
    }

    void Module::publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame) {
        Simulation& state                       = *simulation;
        spectra::sdk::cuda::Frame frame         = output.begin(state.stream.get());
        const spectra::sdk::cuda::Volume volume = frame.volume<"smoke">();
        const std::span<float> density          = volume.field<"density", float>();
        const std::span<float> temperature      = volume.field<"temperature", float>();
        ::cuda::copy_bytes(state.stream, state.current_state.density.values, ::cuda::std::span{density.data(), density.size()});
        ::cuda::copy_bytes(state.stream, state.current_state.temperature.values, ::cuda::std::span{temperature.data(), temperature.size()});
        const spectra::sdk::cuda::MacField velocity = volume.field<"velocity">();
        ::cuda::copy_bytes(state.stream, state.current_state.velocity.x, ::cuda::std::span{velocity.x.data(), velocity.x.size()});
        ::cuda::copy_bytes(state.stream, state.current_state.velocity.y, ::cuda::std::span{velocity.y.data(), velocity.y.size()});
        ::cuda::copy_bytes(state.stream, state.current_state.velocity.z, ::cuda::std::span{velocity.z.data(), velocity.z.size()});
        frame.metric<"step">().upload(state.step_index);
        frame.metric<"time">().upload(state.physical_time);
        frame.commit();
    }
} // namespace physica::examples::smoke
