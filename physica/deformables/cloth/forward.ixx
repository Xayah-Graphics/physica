export module physica.deformables.cloth.forward;

import std;
import physica.deformables.cloth.model;

export namespace physica::deformables::cloth {
    template <class Algorithm, class Value>
    concept ForwardForceAlgorithm = std::constructible_from<Algorithm, const Model<Value>&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Model<Value>& model, const simulation::VectorField<Value>& vector, simulation::VectorField<Value>& vector_output, const simulation::ScalarField<Value>& scalar, const typename Algorithm::Parameters& parameters, typename Algorithm::Cache& cache, typename Algorithm::Workspace& workspace) {
        { algorithm.allocate_parameters(model) } -> std::same_as<typename Algorithm::Parameters>;
        { algorithm.allocate_cache(model) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(model) } -> std::same_as<typename Algorithm::Workspace>;
        algorithm.forward(model, vector, vector, vector, scalar, parameters, vector_output, cache, workspace);
    };

    template <class Algorithm, class Value>
    concept ForwardIntegratorAlgorithm = std::constructible_from<Algorithm, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Model<Value>& model, const simulation::VectorField<Value>& vector, simulation::VectorField<Value>& vector_output, const simulation::ScalarField<Value>& scalar, typename Algorithm::Cache& cache, typename Algorithm::Workspace& workspace) {
        { algorithm.configuration.time_step } -> std::convertible_to<Value>;
        { algorithm.allocate_cache(model) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(model) } -> std::same_as<typename Algorithm::Workspace>;
        algorithm.forward(model, vector, vector, scalar, vector, vector_output, vector_output, cache, workspace);
    };

    template <class Algorithm, class Value>
    concept ForwardConstraintAlgorithm = std::constructible_from<Algorithm, const Model<Value>&, typename Algorithm::Configuration> && requires(const Algorithm& algorithm, const Model<Value>& model, const simulation::VectorField<Value>& vector, simulation::VectorField<Value>& vector_output, const simulation::ScalarField<Value>& scalar, typename Algorithm::Cache& cache, typename Algorithm::Workspace& workspace) {
        { algorithm.allocate_cache(model) } -> std::same_as<typename Algorithm::Cache>;
        { algorithm.allocate_workspace(model) } -> std::same_as<typename Algorithm::Workspace>;
        algorithm.forward(model, vector, vector, vector, vector, scalar, Value{}, vector_output, vector_output, cache, workspace);
    };
} // namespace physica::deformables::cloth
