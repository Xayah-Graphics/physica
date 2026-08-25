module;

#include <physica/cuda.h>

export module physica.fluids.liquid.sph;

import std;
import physica.fluids.liquid.domain;
import physica.fluids.liquid.operators.density;
import physica.fluids.liquid.operators.neighborhood;
import physica.fluids.liquid.operators.sph_dynamics;

export namespace physica::fluids::liquid::sph {
    template <operators::SPHDynamicsAlgorithm Dynamics>
    struct Solver final {
        struct Configuration final {
            typename Dynamics::Configuration dynamics{};
        };

        struct State final {
            operators::ParticleState particles;
            [[no_unique_address]] typename Dynamics::State dynamics;
        };

        struct StateTangent final {
            operators::ParticleStateTangent particles;
            [[no_unique_address]] typename Dynamics::StateTangent dynamics;
        };

        struct StateAdjoint final {
            operators::ParticleStateAdjoint particles;
            [[no_unique_address]] typename Dynamics::StateAdjoint dynamics;
        };

        struct Parameters final {
            operators::ParticleParameters particles;
            typename Dynamics::Parameters dynamics;
        };

        struct ParameterTangent final {
            operators::ParticleParameterTangent particles;
            typename Dynamics::ParameterTangent dynamics;
        };

        struct ParameterAdjoint final {
            operators::ParticleParameterAdjoint particles;
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

        Solver(const Domain& domain, Configuration configuration) : density({}), dynamics(domain, std::move(configuration.dynamics)) {}

        Solver(const Solver&)            = delete;
        Solver& operator=(const Solver&) = delete;
        Solver(Solver&&)                 = delete;
        Solver& operator=(Solver&&)      = delete;

        [[nodiscard]] State allocate_state(const Domain& domain) const {
            State state{
                .particles = {
                    .positions  = domain.allocate_vector_field<float>(domain.configuration.particle_count),
                    .velocities = domain.allocate_vector_field<float>(domain.configuration.particle_count),
                },
                .dynamics = dynamics.allocate_state(domain),
            };
            domain.clear(state.particles.positions);
            domain.clear(state.particles.velocities);
            return state;
        }

        [[nodiscard]] operators::Control allocate_control(const Domain& domain) const {
            operators::Control control{.external_accelerations = domain.allocate_vector_field<float>(domain.configuration.particle_count)};
            domain.clear(control.external_accelerations);
            return control;
        }

        [[nodiscard]] Parameters allocate_parameters(const Domain& domain) const {
            const std::size_t count = domain.configuration.particle_count;
            return {
                .particles = {
                    .masses           = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .rest_densities   = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .viscosities      = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .surface_tensions = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                },
                .dynamics = dynamics.allocate_parameters(domain),
            };
        }

        [[nodiscard]] StepCache allocate_step_cache(const Domain& domain) const {
            return {.neighborhood = neighborhood.allocate_cache(domain), .densities = domain.allocate_scalar_field<float>(domain.configuration.particle_count), .dynamics = dynamics.allocate_cache(domain)};
        }

        [[nodiscard]] Workspace allocate_workspace(const Domain& domain) const {
            return {.neighborhood = neighborhood.allocate_workspace(domain), .dynamics = dynamics.allocate_workspace(domain)};
        }

        [[nodiscard]] StateTangent allocate_state_tangent(const Domain& domain) const {
            StateTangent tangent{
                .particles = {
                    .positions  = domain.allocate_vector_field<float>(domain.configuration.particle_count),
                    .velocities = domain.allocate_vector_field<float>(domain.configuration.particle_count),
                },
                .dynamics = dynamics.allocate_state_tangent(domain),
            };
            domain.clear(tangent.particles.positions);
            domain.clear(tangent.particles.velocities);
            return tangent;
        }

        [[nodiscard]] operators::ControlTangent allocate_control_tangent(const Domain& domain) const {
            operators::ControlTangent tangent{.external_accelerations = domain.allocate_vector_field<float>(domain.configuration.particle_count)};
            domain.clear(tangent.external_accelerations);
            return tangent;
        }

        [[nodiscard]] ParameterTangent allocate_parameter_tangent(const Domain& domain) const {
            const std::size_t count = domain.configuration.particle_count;
            ParameterTangent tangent{
                .particles = {
                    .masses           = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .rest_densities   = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .viscosities      = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .surface_tensions = ::cuda::device_buffer<float>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                },
                .dynamics = dynamics.allocate_parameter_tangent(domain),
            };
            ::cuda::fill_bytes(domain.stream, tangent.particles.masses, 0u);
            ::cuda::fill_bytes(domain.stream, tangent.particles.rest_densities, 0u);
            ::cuda::fill_bytes(domain.stream, tangent.particles.viscosities, 0u);
            ::cuda::fill_bytes(domain.stream, tangent.particles.surface_tensions, 0u);
            return tangent;
        }

        [[nodiscard]] TangentWorkspace allocate_tangent_workspace(const Domain& domain) const {
            return {.densities = domain.allocate_scalar_field<float>(domain.configuration.particle_count), .dynamics = dynamics.allocate_tangent_workspace(domain)};
        }

        [[nodiscard]] StateAdjoint allocate_state_adjoint(const Domain& domain) const {
            StateAdjoint adjoint{
                .particles = {
                    .positions  = domain.allocate_vector_field<double>(domain.configuration.particle_count),
                    .velocities = domain.allocate_vector_field<double>(domain.configuration.particle_count),
                },
                .dynamics = dynamics.allocate_state_adjoint(domain),
            };
            domain.clear(adjoint.particles.positions);
            domain.clear(adjoint.particles.velocities);
            return adjoint;
        }

        [[nodiscard]] operators::ControlAdjoint allocate_control_adjoint(const Domain& domain) const {
            operators::ControlAdjoint adjoint{.external_accelerations = domain.allocate_vector_field<double>(domain.configuration.particle_count)};
            domain.clear(adjoint.external_accelerations);
            return adjoint;
        }

        [[nodiscard]] ParameterAdjoint allocate_parameter_adjoint(const Domain& domain) const {
            const std::size_t count = domain.configuration.particle_count;
            ParameterAdjoint adjoint{
                .particles = {
                    .masses           = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .rest_densities   = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .viscosities      = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                    .surface_tensions = ::cuda::device_buffer<double>{domain.stream, ::cuda::device_default_memory_pool(domain.stream.device()), count, ::cuda::no_init},
                },
                .dynamics = dynamics.allocate_parameter_adjoint(domain),
            };
            ::cuda::fill_bytes(domain.stream, adjoint.particles.masses, 0u);
            ::cuda::fill_bytes(domain.stream, adjoint.particles.rest_densities, 0u);
            ::cuda::fill_bytes(domain.stream, adjoint.particles.viscosities, 0u);
            ::cuda::fill_bytes(domain.stream, adjoint.particles.surface_tensions, 0u);
            return adjoint;
        }

        [[nodiscard]] AdjointWorkspace allocate_adjoint_workspace(const Domain& domain) const {
            return {.densities = domain.allocate_scalar_field<double>(domain.configuration.particle_count), .dynamics = dynamics.allocate_adjoint_workspace(domain)};
        }

        void copy_state(const Domain& domain, const State& source, State& destination) const {
            domain.copy(source.particles.positions, destination.particles.positions);
            domain.copy(source.particles.velocities, destination.particles.velocities);
            destination.particles.step_index = source.particles.step_index;
            dynamics.copy_state(domain, source.dynamics, destination.dynamics);
        }

        void copy_state_tangent(const Domain& domain, const StateTangent& source, StateTangent& destination) const {
            domain.copy(source.particles.positions, destination.particles.positions);
            domain.copy(source.particles.velocities, destination.particles.velocities);
            dynamics.copy_state_tangent(domain, source.dynamics, destination.dynamics);
        }

        void copy_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const {
            domain.copy(source.particles.positions, destination.particles.positions);
            domain.copy(source.particles.velocities, destination.particles.velocities);
            dynamics.copy_state_adjoint(domain, source.dynamics, destination.dynamics);
        }

        void accumulate_state_adjoint(const Domain& domain, const StateAdjoint& source, StateAdjoint& destination) const {
            domain.accumulate(source.particles.positions, destination.particles.positions);
            domain.accumulate(source.particles.velocities, destination.particles.velocities);
            dynamics.accumulate_state_adjoint(domain, source.dynamics, destination.dynamics);
        }

        void forward(const Domain& domain, const State& state, const operators::Control& control, const Parameters& parameters, State& next_state, StepCache& cache, Workspace& workspace) const {
            neighborhood.build(domain, state.particles.step_index, state.particles.positions, cache.neighborhood, workspace.neighborhood);
            density.forward(domain, state.particles.positions, state.particles.positions, parameters.particles, cache.neighborhood, cache.densities);
            dynamics.forward(domain, state.particles, state.dynamics, control, parameters.particles, parameters.dynamics, cache.neighborhood, cache.densities, next_state.particles, next_state.dynamics, cache.dynamics, workspace.dynamics);
        }

        void jvp(const Domain& domain, const State& state, const Parameters& parameters, const StepCache& cache, const StateTangent& state_tangent, const operators::ControlTangent& control_tangent, const ParameterTangent& parameter_tangent, StateTangent& next_state_tangent, TangentWorkspace& workspace) const {
            density.jvp(domain, state.particles.positions, state.particles.positions, state_tangent.particles.positions, parameters.particles, parameter_tangent.particles, cache.neighborhood, workspace.densities);
            dynamics.jvp(domain, state.particles, state.dynamics, parameters.particles, parameters.dynamics, cache.neighborhood, cache.densities, cache.dynamics, state_tangent.particles, state_tangent.dynamics, control_tangent, parameter_tangent.particles, parameter_tangent.dynamics, workspace.densities, next_state_tangent.particles, next_state_tangent.dynamics, workspace.dynamics);
        }

        void vjp(const Domain& domain, const State& state, const Parameters& parameters, const StepCache& cache, const StateAdjoint& next_state_adjoint, StateAdjoint& previous_state_adjoint, operators::ControlAdjoint& control_adjoint, ParameterAdjoint& parameter_adjoint, AdjointWorkspace& workspace) const {
            domain.clear(workspace.densities);
            dynamics.vjp(domain, state.particles, state.dynamics, parameters.particles, parameters.dynamics, cache.neighborhood, cache.densities, cache.dynamics, next_state_adjoint.particles, next_state_adjoint.dynamics, previous_state_adjoint.particles, previous_state_adjoint.dynamics, control_adjoint, parameter_adjoint.particles, parameter_adjoint.dynamics, workspace.densities, workspace.dynamics);
            density.vjp(domain, state.particles.positions, state.particles.positions, parameters.particles, cache.neighborhood, workspace.densities, previous_state_adjoint.particles.positions, parameter_adjoint.particles);
        }

    private:
        [[no_unique_address]] operators::UniformGridNeighborhood neighborhood;
        [[no_unique_address]] operators::CubicSplineDensity density;
        Dynamics dynamics;
    };
} // namespace physica::fluids::liquid::sph
