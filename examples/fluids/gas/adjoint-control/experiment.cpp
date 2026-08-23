module;

#include <cuda/__functional/call_or.h>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/std/span>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#undef max
#undef min

module physica.example.fluids.gas.adjoint_control;

import std;

namespace physica::examples::adjoint_control {
    namespace {
        struct Point final {
            float x{};
            float y{};
            float z{};
        };

        double sum(const std::span<const float> values) { return std::accumulate(values.begin(), values.end(), 0.0); }

        std::vector<Point> load_vertices(const std::filesystem::path& path) {
            std::ifstream input(path);
            std::string line;
            std::size_t vertex_count = 0u;
            while (std::getline(input, line)) {
                if (line.starts_with("element vertex ")) vertex_count = std::stoull(line.substr(15u));
                if (line == "end_header") break;
            }
            std::vector<Point> result(vertex_count);
            float confidence;
            float intensity;
            for (Point& point : result) input >> point.x >> point.y >> point.z >> confidence >> intensity;
            return result;
        }

        std::vector<std::uint8_t> render_front(const std::span<const float> density, const std::uint32_t resolution, const std::uint32_t scale) {
            const std::uint32_t width = resolution * scale;
            const std::uint32_t height = resolution * scale;
            std::vector<float> projection(static_cast<std::size_t>(resolution) * resolution);
            for (std::uint32_t y = 0u; y < resolution; ++y) for (std::uint32_t x = 0u; x < resolution; ++x) {
                float optical_depth = 0.0F;
                for (std::uint32_t z = 0u; z < resolution; ++z) optical_depth += density[x + static_cast<std::size_t>(resolution) * (y + static_cast<std::size_t>(resolution) * z)];
                projection[x + static_cast<std::size_t>(resolution) * y] = optical_depth;
            }
            const float maximum = std::ranges::max(projection);
            std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 3u);
            for (std::uint32_t image_y = 0u; image_y < height; ++image_y) for (std::uint32_t image_x = 0u; image_x < width; ++image_x) {
                const std::uint32_t x = image_x / scale;
                const std::uint32_t y = resolution - 1u - image_y / scale;
                const float normalized = maximum == 0.0F ? 0.0F : std::clamp(projection[x + static_cast<std::size_t>(resolution) * y] / maximum, 0.0F, 1.0F);
                const float smoke = std::pow(normalized, 0.52F);
                const std::size_t index = 3u * (image_x + static_cast<std::size_t>(width) * image_y);
                pixels[index] = static_cast<std::uint8_t>(2.0F + 244.0F * smoke);
                pixels[index + 1u] = static_cast<std::uint8_t>(20.0F + 232.0F * smoke);
                pixels[index + 2u] = static_cast<std::uint8_t>(58.0F + 197.0F * smoke);
            }
            return pixels;
        }

        void write_front(const std::filesystem::path& path, const std::span<const float> density, const std::uint32_t resolution, const std::uint32_t scale) {
            const std::vector<std::uint8_t> pixels = render_front(density, resolution, scale);
            stbi_write_png(path.string().c_str(), resolution * scale, resolution * scale, 3, pixels.data(), resolution * scale * 3u);
        }
    } // namespace

    Experiment::Experiment(ExperimentConfiguration next_configuration, std::filesystem::path next_bunny_path)
        : configuration(std::move(next_configuration)),
          stream{::cuda::devices[0]},
          domain{create_domain_configuration(), stream},
          solver{domain, {
              .diffusion_iterations = 12u,
              .pressure_iterations = 100u,
              .viscosity = 1.0e-5F,
              .density_buoyancy = 1.0F,
          }},
          control{domain, create_control_configuration()},
          objective{domain, {.control_effort_weight = 2.0e-6, .blur_sigma_cells = 0.0F}},
          bunny_path{std::move(next_bunny_path)},
          problem{configuration.step_count, create_state(create_initial_density()), create_keyframes()},
          evaluator{domain, solver, control, objective, problem} {
        stream.sync();
    }

    Experiment::~Experiment() = default;

    void Experiment::verify(const std::filesystem::path& output_directory) {
        std::filesystem::create_directories(output_directory);
        const std::vector<float> initial_density = download_density(problem.initial_state);
        const std::vector<float> target_density = download_density(problem.keyframes.front().target);
        write_density(output_directory / "initial.png", initial_density);
        write_density(output_directory / "target.png", target_density);

        std::vector<double> parameters(control.parameters.values);
        std::vector<double> direction(parameters.size());
        double direction_squared = 0.0;
        for (std::size_t parameter = 0u; parameter < direction.size(); ++parameter) {
            parameters[parameter] = 0.02 * std::sin(0.371 * static_cast<double>(parameter + 1u));
        }
        const fluids::gas::adjoint_control::EvaluationTrace analytic = evaluator.evaluate(parameters, fluids::gas::adjoint_control::EvaluationMode::objective_gradient);
        for (std::size_t parameter = 0u; parameter < direction.size(); ++parameter) {
            direction[parameter] = analytic.reverse->parameter_gradient[parameter];
            direction_squared += direction[parameter] * direction[parameter];
        }
        const double direction_norm = std::sqrt(direction_squared);
        for (double& value : direction) value /= direction_norm;

        fluids::gas::adjoint_control::DerivativeChecker checker{evaluator};
        const fluids::gas::adjoint_control::DirectionalDerivativeCheck derivative = checker.directional(parameters, direction, 5.0e-2);
        std::vector<std::size_t> ranked_components(parameters.size());
        std::iota(ranked_components.begin(), ranked_components.end(), 0u);
        std::ranges::partial_sort(ranked_components, ranked_components.begin() + 3, [&analytic](const std::size_t left, const std::size_t right) { return std::abs(analytic.reverse->parameter_gradient[left]) > std::abs(analytic.reverse->parameter_gradient[right]); });
        const std::array<std::size_t, 3> components{ranked_components[0], ranked_components[1], ranked_components[2]};
        const std::vector<fluids::gas::adjoint_control::ComponentDerivativeCheck> component_checks = checker.components(parameters, components, 5.0e-2);
        const fluids::gas::adjoint_control::EvaluationTrace trace = evaluator.evaluate(parameters, fluids::gas::adjoint_control::EvaluationMode::objective_gradient);
        double maximum_mass_relative_error = 0.0;
        const double initial_mass = sum(initial_density);
        for (const fluids::gas::adjoint_control::State& state : trace.state) maximum_mass_relative_error = std::max(maximum_mass_relative_error, std::abs(sum(download_density(state)) - initial_mass) / initial_mass);

        const std::array<double, 4> quadratic_target{2.0, -3.0, 0.5, 1.5};
        const std::array<double, 4> quadratic_lower{-1.0, -1.0, -1.0, -0.25};
        const std::array<double, 4> quadratic_upper{1.0, 1.0, 1.0, 0.75};
        const std::array<double, 4> quadratic_initial{0.2, 0.5, -0.7, 0.1};
        fluids::gas::adjoint_control::Lbfgsb quadratic_optimizer{{.memory = 4u, .maximum_iterations = 32u, .maximum_evaluations = 128u, .projected_gradient_tolerance = 1.0e-12}, quadratic_initial, quadratic_lower, quadratic_upper};
        while (quadratic_optimizer.request().kind == fluids::gas::adjoint_control::LbfgsbRequestKind::objective_gradient) {
            const fluids::gas::adjoint_control::LbfgsbRequest request = quadratic_optimizer.request();
            double quadratic_objective = 0.0;
            std::array<double, 4> quadratic_gradient;
            for (std::size_t parameter = 0u; parameter < quadratic_target.size(); ++parameter) {
                quadratic_gradient[parameter] = request.parameters[parameter] - quadratic_target[parameter];
                quadratic_objective += 0.5 * quadratic_gradient[parameter] * quadratic_gradient[parameter];
            }
            quadratic_optimizer.submit(quadratic_objective, quadratic_gradient);
        }
        double quadratic_maximum_error = 0.0;
        for (std::size_t parameter = 0u; parameter < quadratic_target.size(); ++parameter) quadratic_maximum_error = std::max(quadratic_maximum_error, std::abs(quadratic_optimizer.parameters[parameter] - std::clamp(quadratic_target[parameter], quadratic_lower[parameter], quadratic_upper[parameter])));

        std::ofstream report(output_directory / "verification.json");
        report << std::setprecision(17)
               << "{\n"
               << "  \"parameter_count\": " << parameters.size() << ",\n"
               << "  \"objective\": " << derivative.objective << ",\n"
               << "  \"finite_difference\": " << derivative.finite_difference << ",\n"
               << "  \"jvp\": " << derivative.jvp << ",\n"
               << "  \"vjp_dot_direction\": " << derivative.vjp_dot_direction << ",\n"
               << "  \"finite_difference_jvp_relative_error\": " << derivative.finite_difference_jvp_relative_error << ",\n"
               << "  \"jvp_vjp_relative_error\": " << derivative.jvp_vjp_relative_error << ",\n"
               << "  \"maximum_mass_relative_error\": " << maximum_mass_relative_error << ",\n"
               << "  \"lbfgsb_maximum_error\": " << quadratic_maximum_error << ",\n"
               << "  \"lbfgsb_stop_reason\": " << static_cast<std::uint32_t>(quadratic_optimizer.stop_reason) << ",\n"
               << "  \"components\": [\n";
        for (std::size_t index = 0u; index < component_checks.size(); ++index) {
            const fluids::gas::adjoint_control::ComponentDerivativeCheck& component = component_checks[index];
            report << "    {\"parameter\": " << component.parameter << ", \"analytic\": " << component.analytic << ", \"finite_difference\": " << component.finite_difference << ", \"relative_error\": " << component.relative_error << "}" << (index + 1u == component_checks.size() ? "\n" : ",\n");
        }
        report << "  ]\n}\n";
        std::println("Adjoint check: FD={:.9e}, JVP={:.9e}, VJP.p={:.9e}", derivative.finite_difference, derivative.jvp, derivative.vjp_dot_direction);
        std::println("Relative errors: FD/JVP={:.3e}, JVP/VJP={:.3e}, mass={:.3e}, L-BFGS-B={:.3e}", derivative.finite_difference_jvp_relative_error, derivative.jvp_vjp_relative_error, maximum_mass_relative_error, quadratic_maximum_error);
        if (derivative.finite_difference_jvp_relative_error > 1.0e-2) throw std::runtime_error("Finite-difference/JVP verification failed");
        if (derivative.jvp_vjp_relative_error > 2.0e-6) throw std::runtime_error("JVP/VJP verification failed");
        if (maximum_mass_relative_error > 2.0e-6) throw std::runtime_error("Mass conservation verification failed");
        if (quadratic_maximum_error > 1.0e-10) throw std::runtime_error("L-BFGS-B verification failed");
    }

    void Experiment::optimize(const std::filesystem::path& output_directory) {
        std::filesystem::create_directories(output_directory / "frames");
        const std::vector<float> initial_density = download_density(problem.initial_state);
        const std::vector<float> target_density = download_density(problem.keyframes.front().target);
        write_density(output_directory / "initial.png", initial_density);
        write_density(output_directory / "target.png", target_density);

        const fluids::gas::adjoint_control::EvaluationTrace uncontrolled = evaluator.evaluate(control.parameters.values, fluids::gas::adjoint_control::EvaluationMode::objective_gradient);
        const std::vector<std::uint8_t> active_parameters = control.active_parameters(0u, configuration.step_count);
        std::vector<double> lower_bounds(control.parameters.lower_bounds.begin(), control.parameters.lower_bounds.end());
        std::vector<double> upper_bounds(control.parameters.upper_bounds.begin(), control.parameters.upper_bounds.end());
        for (std::size_t parameter = 0u; parameter < active_parameters.size(); ++parameter) if (active_parameters[parameter] == 0u) lower_bounds[parameter] = upper_bounds[parameter] = control.parameters.values[parameter];
        fluids::gas::adjoint_control::Lbfgsb optimizer{{
            .memory = 16u,
            .maximum_iterations = configuration.optimizer_iterations,
            .maximum_evaluations = configuration.optimizer_iterations * 12u,
            .maximum_line_search_evaluations = 24u,
            .projected_gradient_tolerance = 1.0e-6,
            .relative_objective_tolerance = 1.0e-10,
        }, control.parameters.values, lower_bounds, upper_bounds};
        fluids::gas::adjoint_control::OptimizationResult result{};
        const auto begin = std::chrono::steady_clock::now();
        while (optimizer.request().kind == fluids::gas::adjoint_control::LbfgsbRequestKind::objective_gradient) {
            const fluids::gas::adjoint_control::LbfgsbRequest request = optimizer.request();
            fluids::gas::adjoint_control::EvaluationTrace trace = evaluator.evaluate(request.parameters, fluids::gas::adjoint_control::EvaluationMode::objective_gradient);
            result.evaluations.push_back({
                .coordinates = {.optimizer_iteration = request.iteration, .objective_evaluation = request.evaluation, .line_search_evaluation = request.line_search_evaluation, .line_search_step = request.step_length},
                .summary = trace.summary,
            });
            std::println("Evaluation {} iteration {} line {}: objective={:.9e}, density={:.9e}, control={:.9e}, |pg|={:.6e}", request.evaluation, request.iteration, request.line_search_evaluation, trace.summary.objective, trace.summary.density_loss, trace.summary.control_effort, trace.summary.projected_gradient_norm);
            optimizer.submit(trace.summary.objective, trace.reverse->parameter_gradient);
        }
        result.parameters = optimizer.parameters;
        result.stop_reason = optimizer.stop_reason;
        result.final_trace.emplace(evaluator.evaluate(result.parameters, fluids::gas::adjoint_control::EvaluationMode::objective_gradient));
        const double optimization_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();

        std::ofstream evaluations(output_directory / "evaluations.csv");
        evaluations << "record,optimizer_iteration,objective_evaluation,line_search_evaluation,line_search_step,objective,density_loss,control_effort,gradient_norm,projected_gradient_norm\n" << std::setprecision(17);
        for (std::size_t record = 0u; record < result.evaluations.size(); ++record) {
            const fluids::gas::adjoint_control::OptimizationEvaluation& evaluation = result.evaluations[record];
            evaluations << record << ',' << evaluation.coordinates.optimizer_iteration << ',' << evaluation.coordinates.objective_evaluation << ',' << evaluation.coordinates.line_search_evaluation << ',' << evaluation.coordinates.line_search_step << ',' << evaluation.summary.objective << ',' << evaluation.summary.density_loss << ',' << evaluation.summary.control_effort << ',' << evaluation.summary.gradient_norm << ',' << evaluation.summary.projected_gradient_norm << '\n';
        }

        write_parameters(output_directory / "parameters.bin", result.parameters);
        std::vector<std::vector<float>> frames;
        frames.reserve(configuration.step_count + configuration.post_step_count + 1u);
        for (std::uint32_t step = 0u; step <= configuration.step_count; ++step) frames.push_back(download_density(result.final_trace->state[step]));

        fluids::gas::adjoint_control::State current = solver.allocate_state(domain);
        fluids::gas::adjoint_control::State next = solver.allocate_state(domain);
        fluids::gas::adjoint_control::DenseControl dense_control = solver.allocate_control(domain);
        fluids::gas::adjoint_control::StepCache cache = solver.allocate_step_cache(domain);
        solver.copy(domain, result.final_trace->state.back(), current);
        control.upload_parameters(domain, result.parameters);
        for (std::uint32_t offset = 0u; offset < configuration.post_step_count; ++offset) {
            const std::uint32_t step = configuration.step_count + offset;
            control.forward(domain, step, dense_control);
            solver.forward(domain, current, dense_control, next, cache);
            frames.push_back(download_density(next));
            std::swap(current, next);
        }

        for (std::uint32_t step = 0u; step < frames.size(); ++step) write_density(output_directory / "frames" / std::format("physical-{:03}.png", step), frames[step]);
        write_sequence(output_directory / "sequence.png", frames);
        const std::vector<float>& final_density = frames[configuration.step_count];
        write_density(output_directory / "final.png", final_density);
        std::vector<float> residual(final_density.size());
        for (std::size_t index = 0u; index < residual.size(); ++index) residual[index] = std::abs(final_density[index] - target_density[index]);
        write_density(output_directory / "residual.png", residual);
        const ShapeMetrics metrics = shape_metrics(final_density, target_density, result.final_trace->state.back());
        const fluids::gas::adjoint_control::EvaluationSummary& final_summary = result.final_trace->summary;
        const double objective_reduction = 1.0 - final_summary.objective / uncontrolled.summary.objective;

        std::ofstream summary(output_directory / "summary.json");
        summary << std::setprecision(17)
                << "{\n"
                << "  \"resolution\": " << configuration.resolution << ",\n"
                << "  \"controlled_step_count\": " << configuration.step_count << ",\n"
                << "  \"post_step_count\": " << configuration.post_step_count << ",\n"
                << "  \"parameter_count\": " << result.parameters.size() << ",\n"
                << "  \"uncontrolled_objective\": " << uncontrolled.summary.objective << ",\n"
                << "  \"objective\": " << final_summary.objective << ",\n"
                << "  \"objective_reduction\": " << objective_reduction << ",\n"
                << "  \"density_loss\": " << final_summary.density_loss << ",\n"
                << "  \"control_effort\": " << final_summary.control_effort << ",\n"
                << "  \"gradient_norm\": " << final_summary.gradient_norm << ",\n"
                << "  \"projected_gradient_norm\": " << final_summary.projected_gradient_norm << ",\n"
                << "  \"relative_l2\": " << metrics.relative_l2 << ",\n"
                << "  \"normalized_cross_correlation\": " << metrics.normalized_cross_correlation << ",\n"
                << "  \"soft_dice\": " << metrics.soft_dice << ",\n"
                << "  \"mass_relative_error\": " << metrics.mass_relative_error << ",\n"
                << "  \"maximum_divergence\": " << metrics.maximum_divergence << ",\n"
                << "  \"rms_divergence\": " << metrics.rms_divergence << ",\n"
                << "  \"maximum_speed\": " << metrics.maximum_speed << ",\n"
                << "  \"dimensionless_maximum_divergence\": " << metrics.dimensionless_maximum_divergence << ",\n"
                << "  \"optimization_evaluations\": " << result.evaluations.size() << ",\n"
                << "  \"optimization_seconds\": " << optimization_seconds << ",\n"
                << "  \"stop_reason\": " << static_cast<std::uint32_t>(result.stop_reason) << "\n"
                << "}\n";

        std::ofstream report(output_directory / "REPORT.md");
        report << "# Adjoint Control Stanford Bunny Reproduction\n\n"
               << "The experiment uses a " << configuration.resolution << "^3 MAC grid, " << configuration.step_count << " controlled steps, " << configuration.post_step_count << " uncontrolled continuation steps, and " << result.parameters.size() << " Gaussian wind parameters.\n\n"
               << "| Metric | Result |\n|---|---:|\n"
               << "| Uncontrolled objective | " << uncontrolled.summary.objective << " |\n"
               << "| Final objective | " << final_summary.objective << " |\n"
               << "| Objective reduction | " << 100.0 * objective_reduction << "% |\n"
               << "| Relative L2 | " << metrics.relative_l2 << " |\n"
               << "| Normalized cross-correlation | " << metrics.normalized_cross_correlation << " |\n"
               << "| Soft Dice | " << metrics.soft_dice << " |\n"
               << "| Mass relative error | " << metrics.mass_relative_error << " |\n"
               << "| Maximum divergence | " << metrics.maximum_divergence << " |\n"
               << "| RMS divergence | " << metrics.rms_divergence << " |\n"
               << "| Maximum speed | " << metrics.maximum_speed << " |\n"
               << "| Dimensionless maximum divergence | " << metrics.dimensionless_maximum_divergence << " |\n"
               << "| Optimization evaluations | " << result.evaluations.size() << " |\n"
               << "| Optimization time | " << optimization_seconds << " s |\n\n"
               << "## Images\n\n![Sequence](sequence.png)\n\n"
               << "| Initial | Target | Controlled result | Absolute residual |\n|---|---|---|---|\n"
               << "| ![](initial.png) | ![](target.png) | ![](final.png) | ![](residual.png) |\n";
        std::println("Bunny final: objective={:.8e}, reduction={:.2f}%, NCC={:.5f}, Dice={:.5f}, L2={:.5f}", final_summary.objective, 100.0 * objective_reduction, metrics.normalized_cross_correlation, metrics.soft_dice, metrics.relative_l2);
    }

    std::pair<std::size_t, double> Experiment::measure_gradient() {
        const auto begin = std::chrono::steady_clock::now();
        const fluids::gas::adjoint_control::EvaluationTrace trace = evaluator.evaluate(control.parameters.values, fluids::gas::adjoint_control::EvaluationMode::objective_gradient);
        stream.sync();
        return {trace.reverse->parameter_gradient.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count()};
    }

    fluids::gas::adjoint_control::DomainConfiguration Experiment::create_domain_configuration() const {
        fluids::gas::adjoint_control::DomainConfiguration result{
            .dimension = fluids::gas::adjoint_control::SpatialDimension::volumetric,
            .resolution = {configuration.resolution, configuration.resolution, configuration.resolution},
            .cell_size = 1.0F / configuration.resolution,
            .time_step = configuration.time_step,
        };
        result.velocity_boundary.x_min.mode = fluids::gas::adjoint_control::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.x_max.mode = fluids::gas::adjoint_control::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_min.mode = fluids::gas::adjoint_control::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_max.mode = fluids::gas::adjoint_control::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.z_min.mode = fluids::gas::adjoint_control::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.z_max.mode = fluids::gas::adjoint_control::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        return result;
    }

    fluids::gas::adjoint_control::ControlConfiguration Experiment::create_control_configuration() const {
        return {
            .lattice = configuration.control_lattice,
            .step_count = configuration.step_count,
            .gaussian_sigma = 0.075F,
            .lower_bound = -18.0,
            .upper_bound = 18.0,
        };
    }

    std::vector<fluids::gas::adjoint_control::Keyframe> Experiment::create_keyframes() {
        std::vector<float> target = create_target_density();
        return {{
            .step = configuration.step_count,
            .target = create_state(target),
            .density_weight = 1.0 / static_cast<double>(target.size()),
            .velocity_weight = 0.0,
        }};
    }

    fluids::gas::adjoint_control::State Experiment::create_state(const std::span<const float> density) {
        fluids::gas::adjoint_control::State state = solver.allocate_state(domain);
        solver.clear(domain, state);
        ::cuda::copy_bytes(stream, density, state.density.values);
        return state;
    }

    std::vector<float> Experiment::create_initial_density() const {
        const std::uint32_t resolution = configuration.resolution;
        std::vector<float> density(static_cast<std::size_t>(resolution) * resolution * resolution);
        const float cell_size = 1.0F / resolution;
        const Point center{0.5F, 0.18F, 0.5F};
        const float radius = 0.14F;
        const float thickness = 0.035F;
        for (std::uint32_t z = 0u; z < resolution; ++z) for (std::uint32_t y = 0u; y < resolution; ++y) for (std::uint32_t x = 0u; x < resolution; ++x) {
            const float px = (x + 0.5F) * cell_size - center.x;
            const float py = (y + 0.5F) * cell_size - center.y;
            const float pz = (z + 0.5F) * cell_size - center.z;
            const float distance = std::sqrt(px * px + py * py + pz * pz);
            const float shell = (distance - radius) / thickness;
            density[x + static_cast<std::size_t>(resolution) * (y + static_cast<std::size_t>(resolution) * z)] = std::exp(-0.5F * shell * shell);
        }
        return density;
    }

    std::vector<float> Experiment::create_target_density() const {
        std::vector<Point> vertices = load_vertices(bunny_path);
        Point minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
        Point maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
        for (const Point point : vertices) {
            minimum.x = std::min(minimum.x, point.x); minimum.y = std::min(minimum.y, point.y); minimum.z = std::min(minimum.z, point.z);
            maximum.x = std::max(maximum.x, point.x); maximum.y = std::max(maximum.y, point.y); maximum.z = std::max(maximum.z, point.z);
        }
        const float scale = std::min({0.58F / (maximum.x - minimum.x), 0.58F / (maximum.y - minimum.y), 0.48F / (maximum.z - minimum.z)});
        const float x_center = 0.5F * (minimum.x + maximum.x);
        const float z_center = 0.5F * (minimum.z + maximum.z);
        for (Point& point : vertices) {
            point.x = 0.5F + scale * (point.x - x_center);
            point.y = 0.32F + scale * (point.y - minimum.y);
            point.z = 0.5F + scale * (point.z - z_center);
        }

        const std::uint32_t resolution = configuration.resolution;
        const float cell_size = 1.0F / resolution;
        const float sigma = 1.15F * cell_size;
        const int radius = static_cast<int>(std::ceil(3.0F * sigma / cell_size));
        std::vector<float> density(static_cast<std::size_t>(resolution) * resolution * resolution);
        for (const Point point : vertices) {
            const int center_x = static_cast<int>(point.x / cell_size);
            const int center_y = static_cast<int>(point.y / cell_size);
            const int center_z = static_cast<int>(point.z / cell_size);
            for (int z = std::max(0, center_z - radius); z <= std::min(static_cast<int>(resolution) - 1, center_z + radius); ++z) for (int y = std::max(0, center_y - radius); y <= std::min(static_cast<int>(resolution) - 1, center_y + radius); ++y) for (int x = std::max(0, center_x - radius); x <= std::min(static_cast<int>(resolution) - 1, center_x + radius); ++x) {
                const float dx = (x + 0.5F) * cell_size - point.x;
                const float dy = (y + 0.5F) * cell_size - point.y;
                const float dz = (z + 0.5F) * cell_size - point.z;
                const float value = std::exp(-0.5F * (dx * dx + dy * dy + dz * dz) / (sigma * sigma));
                const std::size_t index = x + static_cast<std::size_t>(resolution) * (y + static_cast<std::size_t>(resolution) * z);
                density[index] = std::max(density[index], value);
            }
        }
        const std::vector<float> initial = create_initial_density();
        const double mass_scale = sum(initial) / sum(density);
        for (float& value : density) value = static_cast<float>(mass_scale * value);
        return density;
    }

    std::vector<float> Experiment::download_density(const fluids::gas::adjoint_control::State& state) const {
        std::vector<float> result(domain.cell_count);
        ::cuda::copy_bytes(stream, state.density.values, ::cuda::std::span{result.data(), result.size()});
        stream.sync();
        return result;
    }

    std::array<std::vector<float>, 3> Experiment::download_velocity(const fluids::gas::adjoint_control::State& state) const {
        std::array<std::vector<float>, 3> result{
            std::vector<float>(domain.face_counts[0]),
            std::vector<float>(domain.face_counts[1]),
            std::vector<float>(domain.face_counts[2]),
        };
        ::cuda::copy_bytes(stream, state.velocity.x, ::cuda::std::span{result[0].data(), result[0].size()});
        ::cuda::copy_bytes(stream, state.velocity.y, ::cuda::std::span{result[1].data(), result[1].size()});
        ::cuda::copy_bytes(stream, state.velocity.z, ::cuda::std::span{result[2].data(), result[2].size()});
        stream.sync();
        return result;
    }

    ShapeMetrics Experiment::shape_metrics(const std::span<const float> density, const std::span<const float> target, const fluids::gas::adjoint_control::State& state) const {
        double residual_squared = 0.0;
        double target_squared = 0.0;
        double density_squared = 0.0;
        double product = 0.0;
        for (std::size_t index = 0u; index < density.size(); ++index) {
            const double residual = density[index] - target[index];
            residual_squared += residual * residual;
            target_squared += target[index] * target[index];
            density_squared += density[index] * density[index];
            product += density[index] * target[index];
        }
        const std::array<std::vector<float>, 3> velocity = download_velocity(state);
        const std::uint32_t n = configuration.resolution;
        double maximum_divergence = 0.0;
        double divergence_squared = 0.0;
        double maximum_speed = 0.0;
        for (std::uint32_t z = 0u; z < n; ++z) for (std::uint32_t y = 0u; y < n; ++y) for (std::uint32_t x = 0u; x < n; ++x) {
            const double divergence = (velocity[0][x + 1u + static_cast<std::size_t>(n + 1u) * (y + static_cast<std::size_t>(n) * z)] - velocity[0][x + static_cast<std::size_t>(n + 1u) * (y + static_cast<std::size_t>(n) * z)]
                                     + velocity[1][x + static_cast<std::size_t>(n) * (y + 1u + static_cast<std::size_t>(n + 1u) * z)] - velocity[1][x + static_cast<std::size_t>(n) * (y + static_cast<std::size_t>(n + 1u) * z)]
                                     + velocity[2][x + static_cast<std::size_t>(n) * (y + static_cast<std::size_t>(n) * (z + 1u))] - velocity[2][x + static_cast<std::size_t>(n) * (y + static_cast<std::size_t>(n) * z)]) * n;
            maximum_divergence = std::max(maximum_divergence, std::abs(divergence));
            divergence_squared += divergence * divergence;
            const double vx = 0.5 * (velocity[0][x + 1u + static_cast<std::size_t>(n + 1u) * (y + static_cast<std::size_t>(n) * z)] + velocity[0][x + static_cast<std::size_t>(n + 1u) * (y + static_cast<std::size_t>(n) * z)]);
            const double vy = 0.5 * (velocity[1][x + static_cast<std::size_t>(n) * (y + 1u + static_cast<std::size_t>(n + 1u) * z)] + velocity[1][x + static_cast<std::size_t>(n) * (y + static_cast<std::size_t>(n + 1u) * z)]);
            const double vz = 0.5 * (velocity[2][x + static_cast<std::size_t>(n) * (y + static_cast<std::size_t>(n) * (z + 1u))] + velocity[2][x + static_cast<std::size_t>(n) * (y + static_cast<std::size_t>(n) * z)]);
            maximum_speed = std::max(maximum_speed, std::sqrt(vx * vx + vy * vy + vz * vz));
        }
        return {
            .relative_l2 = std::sqrt(residual_squared / target_squared),
            .normalized_cross_correlation = product / std::sqrt(density_squared * target_squared),
            .soft_dice = 2.0 * product / (density_squared + target_squared),
            .mass_relative_error = std::abs(sum(density) - sum(target)) / sum(target),
            .maximum_divergence = maximum_divergence,
            .rms_divergence = std::sqrt(divergence_squared / density.size()),
            .maximum_speed = maximum_speed,
            .dimensionless_maximum_divergence = maximum_divergence / (n * maximum_speed),
        };
    }

    void Experiment::write_density(const std::filesystem::path& path, const std::span<const float> density) const { write_front(path, density, configuration.resolution, 6u); }

    void Experiment::write_sequence(const std::filesystem::path& path, const std::vector<std::vector<float>>& frames) const {
        const std::uint32_t final_step = configuration.step_count + configuration.post_step_count;
        const std::array<std::uint32_t, 6> steps{0u, configuration.step_count / 2u, configuration.step_count, configuration.step_count + configuration.post_step_count / 4u, configuration.step_count + configuration.post_step_count / 2u, final_step};
        const std::uint32_t panel = configuration.resolution * 6u;
        std::vector<std::uint8_t> image(static_cast<std::size_t>(panel) * 6u * panel * 3u);
        for (std::size_t frame = 0u; frame < steps.size(); ++frame) {
            const std::vector<std::uint8_t> pixels = render_front(frames[steps[frame]], configuration.resolution, 6u);
            for (std::uint32_t y = 0u; y < panel; ++y) std::ranges::copy_n(pixels.begin() + static_cast<std::ptrdiff_t>(y * panel * 3u), panel * 3u, image.begin() + static_cast<std::ptrdiff_t>((y * panel * 6u + frame * panel) * 3u));
        }
        stbi_write_png(path.string().c_str(), panel * 6u, panel, 3, image.data(), panel * 6u * 3u);
    }

    void Experiment::write_parameters(const std::filesystem::path& path, const std::span<const double> parameters) const {
        std::ofstream output(path, std::ios::binary);
        const std::uint64_t count = parameters.size();
        output.write(reinterpret_cast<const char*>(&count), sizeof(count));
        output.write(reinterpret_cast<const char*>(parameters.data()), static_cast<std::streamsize>(parameters.size_bytes()));
    }

    void run_verification(const std::string_view bunny_path, const std::string_view output_directory) {
        ExperimentConfiguration configuration;
        configuration.resolution = 14u;
        configuration.step_count = 4u;
        configuration.post_step_count = 0u;
        configuration.control_lattice = {4u, 4u, 4u};
        configuration.optimizer_iterations = 2u;
        Experiment experiment{configuration, std::filesystem::path{bunny_path}};
        experiment.verify(std::filesystem::path{output_directory});
    }

    void run_bunny(const std::string_view bunny_path, const std::string_view output_directory) {
        const ExperimentConfiguration configuration;
        Experiment experiment{configuration, std::filesystem::path{bunny_path}};
        experiment.optimize(std::filesystem::path{output_directory});
    }

    void benchmark(const std::string_view bunny_path, const std::string_view output_directory) {
        const std::filesystem::path asset{bunny_path};
        const std::filesystem::path directory{output_directory};
        std::filesystem::create_directories(directory);
        const std::array lattices{3u, 7u, 12u};
        std::ofstream output(directory / "scaling.csv");
        output << "lattice,parameter_count,gradient_seconds,forward_simulations,reverse_simulations\n" << std::setprecision(17);
        for (const std::uint32_t lattice : lattices) {
            ExperimentConfiguration configuration;
            configuration.resolution = 18u;
            configuration.step_count = 20u;
            configuration.post_step_count = 0u;
            configuration.control_lattice = {lattice, lattice, lattice};
            configuration.optimizer_iterations = 1u;
            Experiment experiment{configuration, asset};
            experiment.measure_gradient();
            const auto [parameters, seconds] = experiment.measure_gradient();
            output << lattice << ',' << parameters << ',' << seconds << ",20,20\n";
            std::println("Lattice {}^3: {} parameters, gradient {:.6f} s", lattice, parameters, seconds);
        }
    }
} // namespace physica::examples::adjoint_control
