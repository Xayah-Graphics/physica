module;

#include <physica/cuda.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#undef max
#undef min

module physica.example.fluids.gas.keyframe_smoke;

import std;

namespace physica::examples::keyframe_smoke {
    namespace {
        struct Point final {
            float x;
            float y;
        };

        struct Segment final {
            Point begin;
            Point end;
        };

        float segment_distance(const Point point, const Segment segment) {
            const float dx = segment.end.x - segment.begin.x;
            const float dy = segment.end.y - segment.begin.y;
            const float denominator = dx * dx + dy * dy;
            const float parameter = std::clamp(((point.x - segment.begin.x) * dx + (point.y - segment.begin.y) * dy) / denominator, 0.0F, 1.0F);
            const float rx = point.x - (segment.begin.x + parameter * dx);
            const float ry = point.y - (segment.begin.y + parameter * dy);
            return std::sqrt(rx * rx + ry * ry);
        }

        void append_polyline(std::vector<Segment>& segments, const std::initializer_list<Point> points) {
            for (auto point = points.begin(); std::next(point) != points.end(); ++point) segments.push_back({.begin = *point, .end = *std::next(point)});
        }

        double sum(std::span<const float> values) {
            return std::accumulate(values.begin(), values.end(), 0.0);
        }

        void write_density_image(const std::filesystem::path& path, const std::span<const float> density, const std::uint32_t width, const std::uint32_t height, const std::uint32_t scale, const float maximum) {
            const std::uint32_t image_width = width * scale;
            const std::uint32_t image_height = height * scale;
            std::vector<std::uint8_t> pixels(static_cast<std::size_t>(image_width) * image_height * 3u);
            for (std::uint32_t image_y = 0u; image_y < image_height; ++image_y) {
                for (std::uint32_t image_x = 0u; image_x < image_width; ++image_x) {
                    const std::uint32_t x = image_x / scale;
                    const std::uint32_t y = height - 1u - image_y / scale;
                    const float value = std::pow(std::clamp(density[x + static_cast<std::size_t>(width) * y] / maximum, 0.0F, 1.0F), 0.62F);
                    const std::size_t index = 3u * (image_x + static_cast<std::size_t>(image_width) * image_y);
                    pixels[index] = static_cast<std::uint8_t>(3.0F + 235.0F * value);
                    pixels[index + 1u] = static_cast<std::uint8_t>(17.0F + 229.0F * value);
                    pixels[index + 2u] = static_cast<std::uint8_t>(58.0F + 194.0F * value);
                }
            }
            stbi_write_png(path.string().c_str(), image_width, image_height, 3, pixels.data(), image_width * 3u);
        }
    } // namespace

    void compose(const std::filesystem::path& results_directory) {
        constexpr std::array letters{'S', 'M', 'O', 'K', 'E'};
        constexpr std::uint32_t letter_resolution = Experiment::resolution;
        constexpr std::uint32_t width = letter_resolution * letters.size();
        constexpr std::uint32_t height = letter_resolution;
        constexpr std::uint32_t dissipation_steps = 90u;
        constexpr float dissipation_rate = 0.65F;
        const std::filesystem::path output_directory = results_directory / "composition";
        std::filesystem::create_directories(output_directory / "frames");

        std::vector<float> combined(static_cast<std::size_t>(width) * height);
        for (std::uint32_t letter_index = 0u; letter_index < letters.size(); ++letter_index) {
            std::vector<float> density(letter_resolution * letter_resolution);
            std::ifstream input(results_directory / "final" / std::string(1u, letters[letter_index]) / "final-density.bin", std::ios::binary);
            input.read(reinterpret_cast<char*>(density.data()), static_cast<std::streamsize>(density.size() * sizeof(float)));
            for (std::uint32_t y = 0u; y < height; ++y) for (std::uint32_t x = 0u; x < letter_resolution; ++x) combined[letter_index * letter_resolution + x + static_cast<std::size_t>(width) * y] = density[x + letter_resolution * y];
        }
        const float initial_maximum = std::ranges::max(combined);
        write_density_image(output_directory / "smoke.png", combined, width, height, 6u, initial_maximum);

        ::cuda::stream stream{::cuda::devices[0]};
        fluids::gas::keyframe_smoke::DomainConfiguration domain_configuration{
            .dimension = fluids::gas::keyframe_smoke::SpatialDimension::planar,
            .resolution = {width, height, 1u},
            .cell_size = Experiment::cell_size,
            .time_step = Experiment::time_step,
        };
        domain_configuration.velocity_boundary.x_min.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        domain_configuration.velocity_boundary.x_max.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        domain_configuration.velocity_boundary.y_min.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        domain_configuration.velocity_boundary.y_max.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::zero_gradient;
        domain_configuration.velocity_boundary.z_min.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::fixed_value;
        domain_configuration.velocity_boundary.z_max.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::fixed_value;
        fluids::gas::keyframe_smoke::Domain domain{std::move(domain_configuration), stream};
        fluids::gas::keyframe_smoke::Solver solver{domain, {
            .diffusion_iterations = 16u,
            .pressure_iterations = 120u,
            .viscosity = 2.0e-5F,
            .density_buoyancy = 0.20F,
            .vorticity_confinement = 0.0F,
            .density_dissipation = dissipation_rate,
        }};
        fluids::gas::keyframe_smoke::State current = solver.allocate_state(domain);
        fluids::gas::keyframe_smoke::State next = solver.allocate_state(domain);
        fluids::gas::keyframe_smoke::DenseControl control = solver.allocate_control(domain);
        fluids::gas::keyframe_smoke::StepCache cache = solver.allocate_step_cache(domain);
        ::cuda::copy_bytes(stream, combined, current.density.values);
        domain.clear(current.velocity);
        solver.clear(domain, control);

        const double initial_mass = sum(combined);
        std::ofstream masses(output_directory / "mass.csv");
        masses << "step,physical_time,mass,expected_mass\n" << std::setprecision(17);
        for (std::uint32_t step = 0u; step <= dissipation_steps; ++step) {
            ::cuda::copy_bytes(stream, current.density.values, ::cuda::std::span{combined.data(), combined.size()});
            stream.sync();
            const double physical_time = step * Experiment::time_step;
            masses << step << ',' << physical_time << ',' << sum(combined) << ',' << initial_mass * std::exp(-dissipation_rate * physical_time) << '\n';
            write_density_image(output_directory / "frames" / std::format("dissipation-{:03}.png", step), combined, width, height, 6u, initial_maximum);
            std::ofstream raw(output_directory / "frames" / std::format("dissipation-{:03}.bin", step), std::ios::binary);
            raw.write(reinterpret_cast<const char*>(combined.data()), static_cast<std::streamsize>(combined.size() * sizeof(float)));
            if (step == dissipation_steps) continue;
            solver.forward(domain, current, control, next, cache);
            std::swap(current, next);
        }
        std::filesystem::copy_file(output_directory / "frames" / std::format("dissipation-{:03}.png", dissipation_steps), output_directory / "dissipated.png", std::filesystem::copy_options::overwrite_existing);
        std::ofstream metadata(output_directory / "metadata.json");
        metadata << "{\n"
                 << "  \"resolution\": [" << width << ", " << height << ", 1],\n"
                 << "  \"physical_step_count\": " << dissipation_steps << ",\n"
                 << "  \"time_step\": " << std::setprecision(9) << Experiment::time_step << ",\n"
                 << "  \"density_dissipation_rate\": " << dissipation_rate << "\n"
                 << "}\n";
        std::println("Composed SMOKE and simulated {} uncontrolled dissipation steps", dissipation_steps);
    }

    Experiment::Experiment(const char next_letter)
        : stream{::cuda::devices[0]},
          domain{create_domain_configuration(), stream},
          solver{domain, {
              .diffusion_iterations = 16u,
              .pressure_iterations = 100u,
              .viscosity = 2.0e-5F,
              .density_buoyancy = 0.65F,
              .vorticity_confinement = 0.0F,
          }},
          control{domain, create_control_configuration(next_letter)},
          objective{domain, {.control_effort_weight = 1.0e-4, .blur_sigma_cells = 4.0F}},
          letter{next_letter},
          problem{
              step_count,
              create_state(create_initial_density()),
              create_keyframes(),
          },
          evaluator{domain, solver, control, objective, problem} {
        stream.sync();
    }

    Experiment::~Experiment() = default;

    void Experiment::optimize(const std::filesystem::path& output_directory) {
        std::filesystem::create_directories(output_directory / "parameters");
        std::filesystem::create_directories(output_directory / "levels");
        std::filesystem::create_directories(output_directory / "frames");
        const std::vector<float> initial_density = download_density(problem.initial_state);
        const std::vector<float> target_density = download_density(problem.keyframes.back().target);
        write_density(output_directory / "initial.png", initial_density);
        write_density(output_directory / "target.png", target_density);

        std::ofstream metadata(output_directory / "metadata.json");
        metadata << "{\n"
                 << "  \"letter\": \"" << letter << "\",\n"
                 << "  \"spatial_dimensions\": 2,\n"
                 << "  \"resolution\": [" << resolution << ", " << resolution << ", 1],\n"
                 << "  \"physical_step_count\": " << step_count << ",\n"
                 << "  \"time_step\": " << std::setprecision(9) << time_step << ",\n"
                 << "  \"control_window_steps\": " << control_window_steps << ",\n"
                 << "  \"parameter_count\": " << control.parameters.values.size() << "\n"
                 << "}\n";
        metadata.close();

        std::ofstream evaluations(output_directory / "evaluations.csv");
        evaluations << "record,continuation_level,optimizer_iteration,objective_evaluation,line_search_evaluation,line_search_step,objective,density_loss,control_effort,gradient_norm,projected_gradient_norm,parameter_file\n";
        evaluations << std::setprecision(17);

        std::vector<double> parameters = control.parameters.values;
        std::vector<std::uint8_t> active_parameters = control.active_parameters(0u, step_count);
        for (std::size_t parameter = 0u; parameter < active_parameters.size(); ++parameter) if (control.parameters.lower_bounds[parameter] == control.parameters.upper_bounds[parameter]) active_parameters[parameter] = 0u;

        const std::array continuation{
            fluids::gas::keyframe_smoke::ContinuationLevel{.blur_sigma_cells = 2.0F, .optimizer = {.memory = 12u, .maximum_iterations = 48u, .maximum_evaluations = 480u, .maximum_line_search_evaluations = 24u, .projected_gradient_tolerance = 2.0e-5, .relative_objective_tolerance = 1.0e-9}},
            fluids::gas::keyframe_smoke::ContinuationLevel{.blur_sigma_cells = 1.0F, .optimizer = {.memory = 12u, .maximum_iterations = 56u, .maximum_evaluations = 560u, .maximum_line_search_evaluations = 24u, .projected_gradient_tolerance = 1.0e-5, .relative_objective_tolerance = 5.0e-10}},
            fluids::gas::keyframe_smoke::ContinuationLevel{.blur_sigma_cells = 0.5F, .optimizer = {.memory = 16u, .maximum_iterations = 72u, .maximum_evaluations = 720u, .maximum_line_search_evaluations = 28u, .projected_gradient_tolerance = 5.0e-6, .relative_objective_tolerance = 2.0e-10}},
            fluids::gas::keyframe_smoke::ContinuationLevel{.blur_sigma_cells = 0.0F, .optimizer = {.memory = 16u, .maximum_iterations = 64u, .maximum_evaluations = 640u, .maximum_line_search_evaluations = 32u, .projected_gradient_tolerance = 2.0e-6, .relative_objective_tolerance = 0.0}},
            fluids::gas::keyframe_smoke::ContinuationLevel{.blur_sigma_cells = 0.0F, .optimizer = {.memory = 16u, .maximum_iterations = 96u, .maximum_evaluations = 960u, .maximum_line_search_evaluations = 36u, .projected_gradient_tolerance = 1.0e-6, .relative_objective_tolerance = 0.0}},
        };

        std::uint64_t record_index = 0u;
        fluids::gas::keyframe_smoke::EvaluationSummary final_summary{};
        std::vector<float> final_density;
        for (std::uint32_t level = 0u; level < continuation.size(); ++level) {
            fluids::gas::keyframe_smoke::OptimizationRunner runner{
                .domain = domain,
                .evaluator = evaluator,
                .objective = objective,
                .control = control,
                .continuation = {continuation[level]},
            };
            fluids::gas::keyframe_smoke::OptimizationResult result = runner.run(parameters, active_parameters, {.continuation_level = level});
            for (const fluids::gas::keyframe_smoke::OptimizationEvaluation& evaluation : result.evaluations) {
                const std::filesystem::path parameter_path = std::filesystem::path{"parameters"} / std::format("evaluation-{:06}.bin", record_index);
                write_parameters(output_directory / parameter_path, evaluation.parameters);
                evaluations << record_index << ','
                            << evaluation.coordinates.continuation_level << ','
                            << evaluation.coordinates.optimizer_iteration << ','
                            << evaluation.coordinates.objective_evaluation << ','
                            << evaluation.coordinates.line_search_evaluation << ','
                            << evaluation.coordinates.line_search_step << ','
                            << evaluation.summary.objective << ','
                            << evaluation.summary.density_loss << ','
                            << evaluation.summary.control_effort << ','
                            << evaluation.summary.gradient_norm << ','
                            << evaluation.summary.projected_gradient_norm << ','
                            << parameter_path.generic_string() << '\n';
                ++record_index;
            }
            evaluations.flush();
            parameters = result.parameters;
            write_parameters(output_directory / std::format("checkpoint-level-{}.bin", level), parameters);
            final_summary = result.final_trace->summary;
            final_density = download_density(result.final_trace->state.back());
            write_density(output_directory / "levels" / std::format("level-{}.png", level), final_density);
            std::println("{} level {}: objective={:.9e}, density={:.9e}, control={:.9e}, evaluations={}, stop={}", letter, level, final_summary.objective, final_summary.density_loss, final_summary.control_effort, result.evaluations.size(), static_cast<std::uint32_t>(result.level_stop_reasons.front()));

            if (level + 1u == continuation.size()) {
                for (std::uint32_t step = 0u; step <= step_count; ++step) {
                    const std::vector<float> density = download_density(result.final_trace->state[step]);
                    write_density(output_directory / "frames" / std::format("physical-{:03}.png", step), density);
                    std::ofstream raw(output_directory / "frames" / std::format("physical-{:03}.bin", step), std::ios::binary);
                    raw.write(reinterpret_cast<const char*>(density.data()), static_cast<std::streamsize>(density.size() * sizeof(float)));
                }
            }
        }
        evaluations.close();

        write_parameters(output_directory / "parameters.bin", parameters);
        write_density(output_directory / "final.png", final_density);
        std::ofstream final_raw(output_directory / "final-density.bin", std::ios::binary);
        final_raw.write(reinterpret_cast<const char*>(final_density.data()), static_cast<std::streamsize>(final_density.size() * sizeof(float)));
        final_raw.close();

        std::vector<float> residual(final_density.size());
        for (std::size_t index = 0u; index < residual.size(); ++index) residual[index] = std::abs(final_density[index] - target_density[index]);
        write_density(output_directory / "residual.png", residual);
        const ShapeMetrics metrics = shape_metrics(final_density, target_density);
        std::ofstream summary(output_directory / "summary.json");
        summary << std::setprecision(17)
                << "{\n"
                << "  \"letter\": \"" << letter << "\",\n"
                << "  \"objective\": " << final_summary.objective << ",\n"
                << "  \"density_loss\": " << final_summary.density_loss << ",\n"
                << "  \"control_effort\": " << final_summary.control_effort << ",\n"
                << "  \"gradient_norm\": " << final_summary.gradient_norm << ",\n"
                << "  \"projected_gradient_norm\": " << final_summary.projected_gradient_norm << ",\n"
                << "  \"relative_l2\": " << metrics.relative_l2 << ",\n"
                << "  \"normalized_cross_correlation\": " << metrics.normalized_cross_correlation << ",\n"
                << "  \"soft_dice\": " << metrics.soft_dice << ",\n"
                << "  \"mass_relative_error\": " << metrics.mass_relative_error << ",\n"
                << "  \"optimization_evaluation_count\": " << record_index << "\n"
                << "}\n";
        summary.close();
        std::println("{} final: relative L2={:.6f}, NCC={:.6f}, soft Dice={:.6f}, mass error={:.3e}", letter, metrics.relative_l2, metrics.normalized_cross_correlation, metrics.soft_dice, metrics.mass_relative_error);
    }

    fluids::gas::keyframe_smoke::DomainConfiguration Experiment::create_domain_configuration() {
        fluids::gas::keyframe_smoke::DomainConfiguration result{
            .dimension = fluids::gas::keyframe_smoke::SpatialDimension::planar,
            .resolution = {resolution, resolution, 1u},
            .cell_size = cell_size,
            .time_step = time_step,
        };
        result.velocity_boundary.x_min.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.x_max.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_min.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.y_max.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::normal_fixed_tangent_zero_gradient;
        result.velocity_boundary.z_min.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::fixed_value;
        result.velocity_boundary.z_max.mode = fluids::gas::keyframe_smoke::VelocityBoundaryMode::fixed_value;
        return result;
    }

    fluids::gas::keyframe_smoke::ControlConfiguration Experiment::create_control_configuration(const char target_letter) {
        fluids::gas::keyframe_smoke::ControlConfiguration result;
        const auto fixed = [](const double value) { return fluids::gas::keyframe_smoke::BoundedValue{.initial = value, .lower = value, .upper = value}; };
        const fluids::gas::keyframe_smoke::BoundedValue free_force{.initial = 0.0, .lower = -4.0, .upper = 4.0};
        const fluids::gas::keyframe_smoke::BoundedValue free_vortex{.initial = 0.0, .lower = -40.0, .upper = 40.0};
        for (std::uint32_t begin_step = 0u; begin_step < step_count; begin_step += control_window_steps) {
            for (std::uint32_t y = 0u; y < 5u; ++y) {
                for (std::uint32_t x = 0u; x < 5u; ++x) {
                    result.winds.push_back({
                        .begin_step = begin_step,
                        .end_step = begin_step + control_window_steps,
                        .width = 0.18F,
                        .center = {fixed(0.10 + 0.20 * x), fixed(0.10 + 0.20 * y), fixed(0.5 * cell_size)},
                        .vector = {free_force, free_force, fixed(0.0)},
                    });
                }
            }
            for (std::uint32_t y = 0u; y < 4u; ++y) {
                for (std::uint32_t x = 0u; x < 4u; ++x) {
                    result.vortices.push_back({
                        .begin_step = begin_step,
                        .end_step = begin_step + control_window_steps,
                        .width = 0.22F,
                        .axis = {0.0F, 0.0F, 1.0F},
                        .center = {fixed(0.20 + 0.20 * x), fixed(0.20 + 0.20 * y), fixed(0.5 * cell_size)},
                        .strength = free_vortex,
                    });
                }
            }
        }
        if (target_letter == 'E') {
            for (std::uint32_t begin_step = 0u; begin_step < step_count; begin_step += control_window_steps) {
                for (std::uint32_t x = 0u; x < 6u; ++x) result.winds.push_back({
                    .begin_step = begin_step,
                    .end_step = begin_step + control_window_steps,
                    .width = 0.075F,
                    .center = {fixed(0.28 + 0.088 * x), fixed(0.55), fixed(0.5 * cell_size)},
                    .vector = {free_force, free_force, fixed(0.0)},
                });
                for (std::uint32_t y = 0u; y < 2u; ++y) {
                    for (std::uint32_t x = 0u; x < 3u; ++x) result.vortices.push_back({
                        .begin_step = begin_step,
                        .end_step = begin_step + control_window_steps,
                        .width = 0.10F,
                        .axis = {0.0F, 0.0F, 1.0F},
                        .center = {fixed(0.38 + 0.12 * x), fixed(0.50 + 0.10 * y), fixed(0.5 * cell_size)},
                        .strength = free_vortex,
                    });
                }
            }
        }
        return result;
    }

    std::vector<fluids::gas::keyframe_smoke::Keyframe> Experiment::create_keyframes() {
        std::vector<fluids::gas::keyframe_smoke::Keyframe> result;
        result.push_back({
            .step = step_count / 3u,
            .target = create_state(create_transport_density(1.0F / 3.0F)),
            .density_weight = 0.25,
            .velocity_weight = 0.0,
            .pseudo = true,
        });
        result.push_back({
            .step = 2u * step_count / 3u,
            .target = create_state(create_transport_density(2.0F / 3.0F)),
            .density_weight = 0.50,
            .velocity_weight = 0.0,
            .pseudo = true,
        });
        if (letter == 'E') result.push_back({
            .step = 5u * step_count / 6u,
            .target = create_state(create_target_density()),
            .density_weight = 3.0,
            .velocity_weight = 0.0,
            .pseudo = true,
        });
        result.push_back({
            .step = step_count,
            .target = create_state(create_target_density()),
            .density_weight = 3.0,
            .velocity_weight = 0.0,
            .pseudo = false,
        });
        return result;
    }

    fluids::gas::keyframe_smoke::State Experiment::create_state(const std::span<const float> density) {
        fluids::gas::keyframe_smoke::State result = solver.allocate_state(domain);
        ::cuda::copy_bytes(stream, density, result.density.values);
        domain.clear(result.velocity);
        return result;
    }

    std::vector<float> Experiment::create_initial_density() const {
        std::vector<float> result(resolution * resolution);
        const Segment capsule{.begin = {0.35F, 0.13F}, .end = {0.65F, 0.13F}};
        for (std::uint32_t y = 0u; y < resolution; ++y) {
            for (std::uint32_t x = 0u; x < resolution; ++x) {
                const Point point{(x + 0.5F) * cell_size, (y + 0.5F) * cell_size};
                const float normalized_distance = segment_distance(point, capsule) / 0.062F;
                result[x + resolution * y] = std::exp(-0.5F * normalized_distance * normalized_distance * normalized_distance * normalized_distance);
            }
        }
        return result;
    }

    std::vector<float> Experiment::create_target_density() const {
        std::vector<Segment> segments;
        if (letter == 'S') append_polyline(segments, {{0.67F, 0.78F}, {0.56F, 0.84F}, {0.40F, 0.82F}, {0.32F, 0.72F}, {0.38F, 0.63F}, {0.60F, 0.58F}, {0.68F, 0.48F}, {0.63F, 0.34F}, {0.50F, 0.28F}, {0.35F, 0.31F}});
        if (letter == 'M') append_polyline(segments, {{0.30F, 0.28F}, {0.32F, 0.82F}, {0.50F, 0.58F}, {0.68F, 0.82F}, {0.70F, 0.28F}});
        if (letter == 'K') {
            append_polyline(segments, {{0.34F, 0.28F}, {0.34F, 0.82F}});
            append_polyline(segments, {{0.70F, 0.82F}, {0.35F, 0.53F}, {0.70F, 0.28F}});
        }
        if (letter == 'E') {
            append_polyline(segments, {{0.68F, 0.82F}, {0.34F, 0.82F}, {0.34F, 0.28F}, {0.70F, 0.28F}});
            append_polyline(segments, {{0.34F, 0.55F}, {0.62F, 0.55F}});
        }
        if (letter == 'O') {
            constexpr std::uint32_t circle_segments = 40u;
            for (std::uint32_t segment = 0u; segment < circle_segments; ++segment) {
                const float first_angle = 2.0F * std::numbers::pi_v<float> * segment / circle_segments;
                const float second_angle = 2.0F * std::numbers::pi_v<float> * (segment + 1u) / circle_segments;
                segments.push_back({
                    .begin = {0.50F + 0.20F * std::cos(first_angle), 0.55F + 0.28F * std::sin(first_angle)},
                    .end = {0.50F + 0.20F * std::cos(second_angle), 0.55F + 0.28F * std::sin(second_angle)},
                });
            }
        }

        std::vector<float> result(resolution * resolution);
        for (std::uint32_t y = 0u; y < resolution; ++y) {
            for (std::uint32_t x = 0u; x < resolution; ++x) {
                const Point point{(x + 0.5F) * cell_size, (y + 0.5F) * cell_size};
                float distance = std::numeric_limits<float>::max();
                for (const Segment segment : segments) distance = std::min(distance, segment_distance(point, segment));
                const float normalized_distance = distance / 0.047F;
                result[x + resolution * y] = std::exp(-0.5F * normalized_distance * normalized_distance * normalized_distance * normalized_distance);
            }
        }
        const std::vector<float> initial = create_initial_density();
        const double scale = sum(initial) / sum(result);
        for (float& value : result) value = static_cast<float>(value * scale);
        return result;
    }

    std::vector<float> Experiment::create_transport_density(const float time) const {
        struct TransportSample final {
            Point position;
            double mass;
        };
        const std::vector<float> initial = create_initial_density();
        const std::vector<float> target = create_target_density();
        std::vector<TransportSample> source_samples;
        std::vector<TransportSample> target_samples;
        for (std::uint32_t y = 0u; y < resolution; ++y) {
            for (std::uint32_t x = 0u; x < resolution; ++x) {
                const std::size_t index = x + resolution * y;
                const Point position{(x + 0.5F) * cell_size, (y + 0.5F) * cell_size};
                if (initial[index] > 1.0e-4F) source_samples.push_back({.position = position, .mass = initial[index]});
                if (target[index] > 1.0e-4F) target_samples.push_back({.position = position, .mass = target[index]});
            }
        }
        double source_mass = 0.0;
        double target_mass = 0.0;
        for (const TransportSample sample : source_samples) source_mass += sample.mass;
        for (const TransportSample sample : target_samples) target_mass += sample.mass;
        const double desired_mass = sum(initial);
        for (TransportSample& sample : source_samples) sample.mass *= desired_mass / source_mass;
        for (TransportSample& sample : target_samples) sample.mass *= desired_mass / target_mass;

        std::vector<double> kernel(source_samples.size() * target_samples.size());
        for (std::size_t source = 0u; source < source_samples.size(); ++source) {
            for (std::size_t destination = 0u; destination < target_samples.size(); ++destination) {
                const double dx = source_samples[source].position.x - target_samples[destination].position.x;
                const double dy = source_samples[source].position.y - target_samples[destination].position.y;
                kernel[source * target_samples.size() + destination] = std::exp(-(dx * dx + dy * dy) / 0.012);
            }
        }
        std::vector<double> left(source_samples.size(), 1.0);
        std::vector<double> right(target_samples.size(), 1.0);
        for (std::uint32_t iteration = 0u; iteration < 160u; ++iteration) {
            for (std::size_t source = 0u; source < source_samples.size(); ++source) {
                double denominator = 0.0;
                for (std::size_t destination = 0u; destination < target_samples.size(); ++destination) denominator += kernel[source * target_samples.size() + destination] * right[destination];
                left[source] = source_samples[source].mass / denominator;
            }
            for (std::size_t destination = 0u; destination < target_samples.size(); ++destination) {
                double denominator = 0.0;
                for (std::size_t source = 0u; source < source_samples.size(); ++source) denominator += kernel[source * target_samples.size() + destination] * left[source];
                right[destination] = target_samples[destination].mass / denominator;
            }
        }

        std::vector<float> result(resolution * resolution, 0.0F);
        for (std::size_t source = 0u; source < source_samples.size(); ++source) {
            for (std::size_t destination = 0u; destination < target_samples.size(); ++destination) {
                const double transported_mass = left[source] * kernel[source * target_samples.size() + destination] * right[destination];
                const float x = ((1.0F - time) * source_samples[source].position.x + time * target_samples[destination].position.x) / cell_size - 0.5F;
                const float y = ((1.0F - time) * source_samples[source].position.y + time * target_samples[destination].position.y) / cell_size - 0.5F;
                const int x0 = static_cast<int>(std::floor(x));
                const int y0 = static_cast<int>(std::floor(y));
                const float tx = x - x0;
                const float ty = y - y0;
                for (int offset_y = 0; offset_y < 2; ++offset_y) {
                    for (int offset_x = 0; offset_x < 2; ++offset_x) {
                        const int sample_x = std::clamp(x0 + offset_x, 0, static_cast<int>(resolution) - 1);
                        const int sample_y = std::clamp(y0 + offset_y, 0, static_cast<int>(resolution) - 1);
                        const float weight_x = offset_x == 0 ? 1.0F - tx : tx;
                        const float weight_y = offset_y == 0 ? 1.0F - ty : ty;
                        result[sample_x + resolution * sample_y] += static_cast<float>(transported_mass * weight_x * weight_y);
                    }
                }
            }
        }
        const double scale = desired_mass / sum(result);
        for (float& value : result) value = static_cast<float>(value * scale);
        return result;
    }

    std::vector<float> Experiment::download_density(const fluids::gas::keyframe_smoke::State& state) const {
        std::vector<float> result(domain.cell_count);
        ::cuda::copy_bytes(stream, state.density.values, ::cuda::std::span{result.data(), result.size()});
        stream.sync();
        return result;
    }

    ShapeMetrics Experiment::shape_metrics(const std::span<const float> density, const std::span<const float> target) const {
        double difference_squared = 0.0;
        double density_squared = 0.0;
        double target_squared = 0.0;
        double dot = 0.0;
        for (std::size_t index = 0u; index < density.size(); ++index) {
            const double difference = density[index] - target[index];
            difference_squared += difference * difference;
            density_squared += density[index] * density[index];
            target_squared += target[index] * target[index];
            dot += density[index] * target[index];
        }
        return {
            .relative_l2 = std::sqrt(difference_squared / target_squared),
            .normalized_cross_correlation = dot / std::sqrt(density_squared * target_squared),
            .soft_dice = 2.0 * dot / (density_squared + target_squared),
            .mass_relative_error = std::abs(sum(density) - sum(target)) / sum(target),
        };
    }

    void Experiment::write_density(const std::filesystem::path& path, const std::span<const float> density) const {
        const float maximum = std::ranges::max(density);
        write_density_image(path, density, resolution, resolution, 8u, maximum);
    }

    void Experiment::write_parameters(const std::filesystem::path& path, const std::span<const double> parameters) const {
        std::ofstream output(path, std::ios::binary);
        const std::uint64_t parameter_count = parameters.size();
        output.write(reinterpret_cast<const char*>(&parameter_count), sizeof(parameter_count));
        output.write(reinterpret_cast<const char*>(parameters.data()), static_cast<std::streamsize>(parameters.size_bytes()));
    }
} // namespace physica::examples::keyframe_smoke
