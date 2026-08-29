module;

#include <physica/cuda.h>

export module physica.fluids.liquid.sph;

import std;
export import physica.fluids.liquid.meshfree;
import physica.fluids.liquid.operators.density;
import physica.fluids.liquid.operators.neighborhood;
export import :dynamics;

export namespace physica::fluids::liquid::sph {
    template <SPHDynamicsAlgorithm Dynamics>
    struct Solver final {
        struct Configuration final {
            typename Dynamics::Configuration dynamics{};
        };

        struct State final {
            ParticleState particles;
            [[no_unique_address]] typename Dynamics::State dynamics;
        };

        struct StateTangent final {
            ParticleStateTangent particles;
            [[no_unique_address]] typename Dynamics::StateTangent dynamics;
        };

        struct StateAdjoint final {
            ParticleStateAdjoint particles;
            [[no_unique_address]] typename Dynamics::StateAdjoint dynamics;
        };

        struct Parameters final {
            ParticleParameters particles;
            typename Dynamics::Parameters dynamics;
        };

        struct ParameterTangent final {
            ParticleParameterTangent particles;
            typename Dynamics::ParameterTangent dynamics;
        };

        struct ParameterAdjoint final {
            ParticleParameterAdjoint particles;
            typename Dynamics::ParameterAdjoint dynamics;
        };

        struct StepCache final {
            operators::Neighborhood neighborhood;
            ScalarField<float> densities;
            typename Dynamics::Cache dynamics;
        };

        struct Workspace final {
            operators::UniformGridNeighborhood::Workspace neighborhood;
            [[no_unique_address]] typename Dynamics::Workspace dynamics;
        };

        struct TangentWorkspace final {
            ScalarField<float> densities;
            typename Dynamics::TangentWorkspace dynamics;
        };

        struct AdjointWorkspace final {
            ScalarField<double> densities;
            typename Dynamics::AdjointWorkspace dynamics;
        };

        Solver(const meshfree::Model& model, Configuration configuration) : density({}), dynamics(model, std::move(configuration.dynamics)) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state(const meshfree::Model& model) const {
            State state{
                .particles =
                    {
                        .positions  = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
                        .velocities = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
                    },
                .dynamics = dynamics.allocate_state(model),
            };
            model.fields.clear(state.particles.positions);
            model.fields.clear(state.particles.velocities);
            return state;
        }

        [[nodiscard]] Control allocate_control(const meshfree::Model& model) const {
            Control control{.external_accelerations = model.fields.allocate_vector_field<float>(model.configuration.particle_count)};
            model.fields.clear(control.external_accelerations);
            return control;
        }

        [[nodiscard]] Parameters allocate_parameters(const meshfree::Model& model) const {
            const std::size_t count = model.configuration.particle_count;
            return {
                .particles =
                    {
                        .masses           = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .rest_densities   = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .viscosities      = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .surface_tensions = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                    },
                .dynamics = dynamics.allocate_parameters(model),
            };
        }

        [[nodiscard]] StepCache allocate_step_cache(const meshfree::Model& model) const {
            return {.neighborhood = neighborhood.allocate_cache(model), .densities = model.fields.allocate_scalar_field<float>(model.configuration.particle_count), .dynamics = dynamics.allocate_cache(model)};
        }

        [[nodiscard]] Workspace allocate_workspace(const meshfree::Model& model) const {
            return {.neighborhood = neighborhood.allocate_workspace(model), .dynamics = dynamics.allocate_workspace(model)};
        }

        [[nodiscard]] StateTangent allocate_state_tangent(const meshfree::Model& model) const {
            StateTangent tangent{
                .particles =
                    {
                        .positions  = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
                        .velocities = model.fields.allocate_vector_field<float>(model.configuration.particle_count),
                    },
                .dynamics = dynamics.allocate_state_tangent(model),
            };
            model.fields.clear(tangent.particles.positions);
            model.fields.clear(tangent.particles.velocities);
            return tangent;
        }

        [[nodiscard]] ControlTangent allocate_control_tangent(const meshfree::Model& model) const {
            ControlTangent tangent{.external_accelerations = model.fields.allocate_vector_field<float>(model.configuration.particle_count)};
            model.fields.clear(tangent.external_accelerations);
            return tangent;
        }

        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const meshfree::Model& model) const {
            const std::size_t count = model.configuration.particle_count;
            ParameterTangent tangent{
                .particles =
                    {
                        .masses           = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .rest_densities   = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .viscosities      = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .surface_tensions = ::cuda::device_buffer<float>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                    },
                .dynamics = dynamics.allocate_parameter_tangent(model),
            };
            ::cuda::fill_bytes(model.fields.stream, tangent.particles.masses, 0u);
            ::cuda::fill_bytes(model.fields.stream, tangent.particles.rest_densities, 0u);
            ::cuda::fill_bytes(model.fields.stream, tangent.particles.viscosities, 0u);
            ::cuda::fill_bytes(model.fields.stream, tangent.particles.surface_tensions, 0u);
            return tangent;
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const meshfree::Model& model) const {
            return {.densities = model.fields.allocate_scalar_field<float>(model.configuration.particle_count), .dynamics = dynamics.allocate_tangent_workspace(model)};
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint(const meshfree::Model& model) const {
            StateAdjoint adjoint{
                .particles =
                    {
                        .positions  = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
                        .velocities = model.fields.allocate_vector_field<double>(model.configuration.particle_count),
                    },
                .dynamics = dynamics.allocate_state_adjoint(model),
            };
            model.fields.clear(adjoint.particles.positions);
            model.fields.clear(adjoint.particles.velocities);
            return adjoint;
        }

        [[nodiscard]] ControlAdjoint allocate_control_adjoint(const meshfree::Model& model) const {
            ControlAdjoint adjoint{.external_accelerations = model.fields.allocate_vector_field<double>(model.configuration.particle_count)};
            model.fields.clear(adjoint.external_accelerations);
            return adjoint;
        }

        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const meshfree::Model& model) const {
            const std::size_t count = model.configuration.particle_count;
            ParameterAdjoint adjoint{
                .particles =
                    {
                        .masses           = ::cuda::device_buffer<double>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .rest_densities   = ::cuda::device_buffer<double>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .viscosities      = ::cuda::device_buffer<double>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                        .surface_tensions = ::cuda::device_buffer<double>{model.fields.stream, ::cuda::device_default_memory_pool(model.fields.stream.device()), count, ::cuda::no_init},
                    },
                .dynamics = dynamics.allocate_parameter_adjoint(model),
            };
            ::cuda::fill_bytes(model.fields.stream, adjoint.particles.masses, 0u);
            ::cuda::fill_bytes(model.fields.stream, adjoint.particles.rest_densities, 0u);
            ::cuda::fill_bytes(model.fields.stream, adjoint.particles.viscosities, 0u);
            ::cuda::fill_bytes(model.fields.stream, adjoint.particles.surface_tensions, 0u);
            return adjoint;
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const meshfree::Model& model) const {
            return {.densities = model.fields.allocate_scalar_field<double>(model.configuration.particle_count), .dynamics = dynamics.allocate_adjoint_workspace(model)};
        }

        void copy_state(const meshfree::Model& model, const State& source, State& destination) const {
            model.fields.copy(source.particles.positions, destination.particles.positions);
            model.fields.copy(source.particles.velocities, destination.particles.velocities);
            destination.particles.step_index = source.particles.step_index;
            dynamics.copy_state(model, source.dynamics, destination.dynamics);
        }

        void copy_state_tangent(const meshfree::Model& model, const StateTangent& source, StateTangent& destination) const {
            model.fields.copy(source.particles.positions, destination.particles.positions);
            model.fields.copy(source.particles.velocities, destination.particles.velocities);
            dynamics.copy_state_tangent(model, source.dynamics, destination.dynamics);
        }

        void copy_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
            model.fields.copy(source.particles.positions, destination.particles.positions);
            model.fields.copy(source.particles.velocities, destination.particles.velocities);
            dynamics.copy_state_adjoint(model, source.dynamics, destination.dynamics);
        }

        void accumulate_state_adjoint(const meshfree::Model& model, const StateAdjoint& source, StateAdjoint& destination) const {
            model.fields.accumulate(source.particles.positions, destination.particles.positions);
            model.fields.accumulate(source.particles.velocities, destination.particles.velocities);
            dynamics.accumulate_state_adjoint(model, source.dynamics, destination.dynamics);
        }

        void forward(const meshfree::Model& model, const State& state, const Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
            neighborhood.build(model, state.particles.step_index, state.particles.positions, cache.neighborhood, workspace.neighborhood);
            density.forward(model, state.particles.positions, state.particles.positions, parameters.particles, cache.neighborhood, cache.densities);
            dynamics.forward(model, state.particles, state.dynamics, control, parameters.particles, parameters.dynamics, cache.neighborhood, cache.densities, next_state.particles, next_state.dynamics, cache.dynamics, workspace.dynamics);
        }

        void jvp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateTangent& state_tangent, const ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
            density.jvp(model, state.particles.positions, state.particles.positions, state_tangent.particles.positions, parameters.particles, parameter_tangent.particles, cache.neighborhood, workspace.densities);
            dynamics.jvp(model, state.particles, state.dynamics, parameters.particles, parameters.dynamics, cache.neighborhood, cache.densities, cache.dynamics, state_tangent.particles, state_tangent.dynamics, control_tangent, parameter_tangent.particles, parameter_tangent.dynamics, workspace.densities, next_state_tangent.particles, next_state_tangent.dynamics, workspace.dynamics);
        }

        void vjp(const meshfree::Model& model, const State& state, const Parameters& parameters, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
            model.fields.clear(workspace.densities);
            dynamics.vjp(model, state.particles, state.dynamics, parameters.particles, parameters.dynamics, cache.neighborhood, cache.densities, cache.dynamics, next_state_adjoint.particles, next_state_adjoint.dynamics, previous_state_adjoint.particles, previous_state_adjoint.dynamics, control_adjoint, parameter_adjoint.particles, parameter_adjoint.dynamics, workspace.densities, workspace.dynamics);
            density.vjp(model, state.particles.positions, state.particles.positions, parameters.particles, cache.neighborhood, workspace.densities, previous_state_adjoint.particles.positions, parameter_adjoint.particles);
        }

    private:
        [[no_unique_address]] operators::UniformGridNeighborhood neighborhood;
        [[no_unique_address]] operators::CubicSplineDensity density;
        Dynamics dynamics;
    };
} // namespace physica::fluids::liquid::sph
