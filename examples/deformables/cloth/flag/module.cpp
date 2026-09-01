module;

#include "module_kernels.h"
#include "simulation_kernels.h"
#include <cuda_runtime_api.h>
#include <physica/cuda.h>

module physica.example.deformables.cloth.flag.module;

import std;
import physica.deformables.cloth.constraints.fixed_position;
import physica.deformables.cloth.integrators.semi_implicit_euler;
import physica.deformables.cloth.model;
import physica.deformables.cloth.operators.mass_spring;
import physica.deformables.cloth.solvers.explicit_dynamics;
import physica.deformables.cloth.topology;
import spectra.sdk;
import spectra.sdk.cuda;

namespace physica::examples::cloth_flag {
    struct Configuration final {
        std::vector<Vector3<float>> rest_positions{};
        std::vector<deformables::cloth::Triangle> triangles{};
        std::vector<deformables::cloth::TriangleMaterialCoordinates<float>> material_coordinates{};
        std::vector<std::uint32_t> anchors{};
        Vector3<float> gravity{};
        Vector3<float> wind{};
        float particle_mass{};
        float stretch_stiffness{};
        float stretch_damping{};
        float bending_stiffness{};
        float bending_damping{};
        float gust_strength{};
        float gust_frequency{};
        float air_density{};
        float drag_coefficient{};
        float step_seconds{};
        std::uint32_t substeps{};
    };

    struct Simulation final {
        ::cuda::stream stream;
        const Configuration configuration;
        deformables::cloth::Model<float> model;

    private:
        deformables::cloth::solvers::explicit_dynamics::Solver<float, deformables::cloth::operators::MassSpringForce, deformables::cloth::integrators::SemiImplicitEuler, deformables::cloth::constraints::FixedPositionConstraint> solver;

    public:
        deformables::cloth::State<float> current_state;
        decltype(solver)::Parameters parameters;
        decltype(solver)::StepCache step_cache;
        std::uint64_t step_index{};
        double physical_time{};

        explicit Simulation(Configuration configuration);

        Simulation(const Simulation&)            = delete;
        Simulation& operator=(const Simulation&) = delete;
        Simulation(Simulation&&)                 = delete;
        Simulation& operator=(Simulation&&)      = delete;

        void reset();
        void step();

    private:
        deformables::cloth::State<float> next_state;
        deformables::cloth::Control<float> control;
        decltype(solver)::Workspace workspace;

        [[nodiscard]] static deformables::cloth::ModelConfiguration<float> model_configuration(const Configuration& configuration);
        [[nodiscard]] static deformables::cloth::constraints::FixedPositionConstraint::Configuration constraint_configuration(const Configuration& configuration);
    };

    Simulation::Simulation(Configuration source) : stream{::cuda::devices[0]}, configuration(std::move(source)), model(model_configuration(configuration), stream), solver(model, {.force = {.gravity = configuration.gravity}, .integrator = {.time_step = configuration.step_seconds / static_cast<float>(configuration.substeps)}, .constraint = constraint_configuration(configuration)}), current_state(solver.allocate_state(model)), parameters(solver.allocate_parameters(model)), step_cache(solver.allocate_step_cache(model)), next_state(solver.allocate_state(model)), control(solver.allocate_control(model)), workspace(solver.allocate_workspace(model)) {
        const std::vector<float> masses(model.particle_count, configuration.particle_mass);
        const std::vector<float> stretch_stiffnesses(parameters.force.stretch.stiffnesses.values.size(), configuration.stretch_stiffness);
        const std::vector<float> stretch_dampings(parameters.force.stretch.dampings.values.size(), configuration.stretch_damping);
        const std::vector<float> bending_stiffnesses(parameters.force.bending.stiffnesses.values.size(), configuration.bending_stiffness);
        const std::vector<float> bending_dampings(parameters.force.bending.dampings.values.size(), configuration.bending_damping);
        ::cuda::copy_bytes(stream, masses, parameters.masses.values);
        ::cuda::copy_bytes(stream, stretch_stiffnesses, parameters.force.stretch.stiffnesses.values);
        ::cuda::copy_bytes(stream, stretch_dampings, parameters.force.stretch.dampings.values);
        ::cuda::copy_bytes(stream, bending_stiffnesses, parameters.force.bending.stiffnesses.values);
        ::cuda::copy_bytes(stream, bending_dampings, parameters.force.bending.dampings.values);
        stream.sync();
        reset();
    }

    void Simulation::reset() {
        simulation::upload(stream, model.configuration.rest_positions, current_state.positions);
        simulation::clear(stream, current_state.velocities);
        simulation::upload(stream, model.configuration.rest_positions, next_state.positions);
        simulation::clear(stream, next_state.velocities);
        simulation::clear(stream, control.external_forces);
        simulation::clear(stream, step_cache.forces);
        step_index    = 0u;
        physical_time = 0.0;
    }

    void Simulation::step() {
        const float substep_seconds = configuration.step_seconds / static_cast<float>(configuration.substeps);
        for (std::uint32_t substep = 0u; substep != configuration.substeps; ++substep) {
            simulation_cuda::write_aerodynamic_forces(stream, static_cast<std::uint32_t>(model.particle_count), static_cast<std::uint32_t>(model.configuration.triangles.size()), step_index * configuration.substeps + substep, substep_seconds, {configuration.wind.x, configuration.wind.y, configuration.wind.z, configuration.gust_strength, configuration.gust_frequency, configuration.air_density, configuration.drag_coefficient}, model.topology.device.triangles.first.values.data(), model.topology.device.triangles.second.values.data(), model.topology.device.triangles.third.values.data(), current_state.positions.x.data(), current_state.positions.y.data(), current_state.positions.z.data(), current_state.velocities.x.data(), current_state.velocities.y.data(), current_state.velocities.z.data(), control.external_forces.x.data(), control.external_forces.y.data(), control.external_forces.z.data());
            solver.forward(model, current_state, control, parameters, next_state, step_cache, workspace);
            std::swap(current_state, next_state);
        }
        ++step_index;
        physical_time = static_cast<double>(step_index) * configuration.step_seconds;
    }

    deformables::cloth::ModelConfiguration<float> Simulation::model_configuration(const Configuration& source) {
        return {source.rest_positions, source.triangles, source.material_coordinates};
    }

    deformables::cloth::constraints::FixedPositionConstraint::Configuration Simulation::constraint_configuration(const Configuration& source) {
        deformables::cloth::constraints::FixedPositionConstraint::Configuration result{};
        result.anchors.reserve(source.anchors.size());
        for (const std::uint32_t particle : source.anchors) result.anchors.emplace_back(particle, source.rest_positions[particle]);
        return result;
    }

    Module::Module(const Settings source, const std::filesystem::path&, const spectra::sdk::SceneInputs& inputs) : settings(source), simulation(std::make_unique<Simulation>(configuration(settings, inputs))) {}

    Module::~Module() = default;

    void Module::setup(spectra::sdk::cuda::Setup& setup) {
        const std::uint32_t particle_count = static_cast<std::uint32_t>(simulation->model.particle_count);
        [[maybe_unused]] const auto surface = setup.mesh<"surface">(particle_count, 0u);
        setup.mesh_field<"velocity">(particle_count);
        setup.mesh_field<"force">(particle_count);
        setup.mesh_field<"strain">(particle_count);

        const spectra::sdk::cuda::IndexedPointsSetup pins = setup.indexed_points<"pins">(static_cast<std::uint32_t>(simulation->configuration.anchors.size()));
        cudaMemcpy(pins.indices.data(), simulation->configuration.anchors.data(), pins.indices.size_bytes(), cudaMemcpyHostToDevice);

        std::vector<spectra::sdk::UInt2> stretch_constraints{};
        stretch_constraints.reserve(simulation->model.topology.edges.size());
        for (const deformables::cloth::Edge edge : simulation->model.topology.edges) stretch_constraints.emplace_back(edge.first, edge.second);
        const spectra::sdk::cuda::IndexedSegmentsSetup stretch = setup.indexed_segments<"stretch-constraints">(static_cast<std::uint32_t>(stretch_constraints.size()));
        cudaMemcpy(stretch.indices.data(), stretch_constraints.data(), stretch.indices.size_bytes(), cudaMemcpyHostToDevice);

        std::vector<spectra::sdk::UInt2> bend_constraints{};
        bend_constraints.reserve(simulation->model.topology.hinges.size());
        for (const deformables::cloth::Hinge hinge : simulation->model.topology.hinges) bend_constraints.emplace_back(hinge.first_opposite, hinge.second_opposite);
        const spectra::sdk::cuda::IndexedSegmentsSetup bend = setup.indexed_segments<"bend-constraints">(static_cast<std::uint32_t>(bend_constraints.size()));
        cudaMemcpy(bend.indices.data(), bend_constraints.data(), bend.indices.size_bytes(), cudaMemcpyHostToDevice);
    }

    void Module::reset(const std::uint64_t) {
        simulation->reset();
    }

    void Module::step(const double) {
        simulation->step();
    }

    void Module::publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame) {
        spectra::sdk::cuda::Frame frame        = output.begin(simulation->stream.get());
        const spectra::sdk::cuda::Mesh surface = frame.mesh<"surface">();
        const std::uint32_t particle_count     = static_cast<std::uint32_t>(simulation->model.particle_count);
        module_cuda::write_vectors(simulation->stream, particle_count, simulation->current_state.positions.x.data(), simulation->current_state.positions.y.data(), simulation->current_state.positions.z.data(), surface.positions.data());
        if (output.requested<"velocity">()) {
            const std::span<spectra::sdk::Float3> velocity = frame.mesh_field<"velocity", spectra::sdk::Float3>(particle_count);
            module_cuda::write_vectors(simulation->stream, particle_count, simulation->current_state.velocities.x.data(), simulation->current_state.velocities.y.data(), simulation->current_state.velocities.z.data(), velocity.data());
        }
        if (output.requested<"force">()) {
            const std::span<spectra::sdk::Float3> force = frame.mesh_field<"force", spectra::sdk::Float3>(particle_count);
            module_cuda::write_vectors(simulation->stream, particle_count, simulation->step_cache.forces.x.data(), simulation->step_cache.forces.y.data(), simulation->step_cache.forces.z.data(), force.data());
        }
        if (output.requested<"strain">()) {
            const std::span<float> strain = frame.mesh_field<"strain", float>(particle_count);
            const deformables::cloth::DeviceEdgeTopology& edges = simulation->model.topology.device.edges;
            module_cuda::write_strain(simulation->stream, particle_count, static_cast<std::uint32_t>(simulation->model.topology.edges.size()), edges.first.values.data(), edges.second.values.data(), simulation->parameters.force.stretch.rest_lengths.values.data(), simulation->current_state.positions.x.data(), simulation->current_state.positions.y.data(), simulation->current_state.positions.z.data(), strain.data());
        }
        if (output.requested<"pins">()) frame.indexed_points<"pins">(static_cast<std::uint32_t>(simulation->configuration.anchors.size()));
        if (output.requested<"stretch-constraints">()) frame.indexed_segments<"stretch-constraints">(static_cast<std::uint32_t>(simulation->model.topology.edges.size()));
        if (output.requested<"bend-constraints">()) frame.indexed_segments<"bend-constraints">(static_cast<std::uint32_t>(simulation->model.topology.hinges.size()));
        frame.metric<"step">().upload(simulation->step_index);
        frame.metric<"time">().upload(simulation->physical_time);
        frame.commit();
    }

    Configuration Module::configuration(const Settings& source, const spectra::sdk::SceneInputs& inputs) {
        const spectra::sdk::MeshInput& mesh = *std::ranges::find(inputs.meshes, std::string_view{"cloth"}, &spectra::sdk::MeshInput::id);
        const spectra::sdk::IndexSelectionInput& pins = *std::ranges::find(mesh.selections, std::string_view{"Pins"}, &spectra::sdk::IndexSelectionInput::id);
        Configuration result{
            .rest_positions       = std::vector<Vector3<float>>(mesh.positions.size()),
            .triangles            = std::vector<deformables::cloth::Triangle>(mesh.indices.size() / 3u),
            .material_coordinates = std::vector<deformables::cloth::TriangleMaterialCoordinates<float>>(mesh.indices.size() / 3u),
            .anchors              = {pins.indices.begin(), pins.indices.end()},
            .gravity              = {source.gravity.x, source.gravity.y, source.gravity.z},
            .wind                 = {source.wind.x, source.wind.y, source.wind.z},
            .particle_mass        = source.particle_mass,
            .stretch_stiffness    = source.stretch_stiffness,
            .stretch_damping      = source.stretch_damping,
            .bending_stiffness    = source.bending_stiffness,
            .bending_damping      = source.bending_damping,
            .gust_strength        = source.gust_strength,
            .gust_frequency       = source.gust_frequency,
            .air_density          = source.air_density,
            .drag_coefficient     = source.drag_coefficient,
            .step_seconds         = static_cast<float>(inputs.step_seconds),
            .substeps             = source.substeps,
        };
        for (std::size_t particle = 0u; particle != mesh.positions.size(); ++particle) result.rest_positions[particle] = {mesh.positions[particle].x, mesh.positions[particle].y, mesh.positions[particle].z};
        for (std::size_t triangle = 0u; triangle != result.triangles.size(); ++triangle) {
            const std::uint32_t first  = mesh.indices[triangle * 3u];
            const std::uint32_t second = mesh.indices[triangle * 3u + 1u];
            const std::uint32_t third  = mesh.indices[triangle * 3u + 2u];
            result.triangles[triangle] = {first, second, third};
            result.material_coordinates[triangle] = {
                {mesh.texture_coordinates[first].x, mesh.texture_coordinates[first].y},
                {mesh.texture_coordinates[second].x, mesh.texture_coordinates[second].y},
                {mesh.texture_coordinates[third].x, mesh.texture_coordinates[third].y},
            };
        }
        return result;
    }
} // namespace physica::examples::cloth_flag
