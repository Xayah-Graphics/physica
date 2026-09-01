module;

#include <spectra/sdk/cuda_types.h>

export module physica.example.deformables.cloth.flag.module;

import std;
import spectra.sdk;
import spectra.sdk.cuda;

export namespace physica::examples::cloth_flag {
    struct Configuration;
    struct Simulation;

    struct Settings final {
        float particle_mass{0.0125F};
        float stretch_stiffness{420.0F};
        float stretch_damping{1.4F};
        float bending_stiffness{4.0F};
        float bending_damping{0.3F};
        spectra::sdk::Float3 gravity{0.0F, 0.0F, -9.81F};
        spectra::sdk::Float3 wind{3.2F, -0.25F, 0.35F};
        float gust_strength{0.3F};
        float gust_frequency{0.8F};
        float air_density{1.225F};
        float drag_coefficient{1.0F};
        std::uint32_t substeps{8u};
    };

    struct Module final {
        Settings settings;

        static constexpr auto recreate = spectra::sdk::ParameterApplication::Recreate;
        static constexpr auto description = spectra::sdk::describe(
            "physica.example.deformables.cloth.flag",
            spectra::sdk::parameter<"particleMass", &Settings::particle_mass>("Particle Mass", "kg", {.minimum = 0.0001, .maximum = 0.1, .step = 0.0001, .application = recreate, .section = "Mass"}),
            spectra::sdk::parameter<"stretchStiffness", &Settings::stretch_stiffness>("Stretch Stiffness", "N/m", {.minimum = 1.0, .maximum = 2000.0, .step = 1.0, .application = recreate, .section = "Material"}),
            spectra::sdk::parameter<"stretchDamping", &Settings::stretch_damping>("Stretch Damping", "N s/m", {.minimum = 0.0, .maximum = 10.0, .step = 0.05, .application = recreate, .section = "Material"}),
            spectra::sdk::parameter<"bendingStiffness", &Settings::bending_stiffness>("Bending Stiffness", "N/m", {.minimum = 0.0, .maximum = 100.0, .step = 0.1, .application = recreate, .section = "Material"}),
            spectra::sdk::parameter<"bendingDamping", &Settings::bending_damping>("Bending Damping", "N s/m", {.minimum = 0.0, .maximum = 10.0, .step = 0.05, .application = recreate, .section = "Material"}),
            spectra::sdk::parameter<"gravity", &Settings::gravity>("Gravity", "m/s²", {.minimum = -20.0, .maximum = 20.0, .step = 0.01, .application = recreate, .section = "Forces"}),
            spectra::sdk::parameter<"wind", &Settings::wind>("Wind", "m/s", {.minimum = -20.0, .maximum = 20.0, .step = 0.05, .application = recreate, .section = "Forces"}),
            spectra::sdk::parameter<"gustStrength", &Settings::gust_strength>("Gust Strength", {}, {.minimum = 0.0, .maximum = 1.0, .step = 0.01, .application = recreate, .section = "Forces"}),
            spectra::sdk::parameter<"gustFrequency", &Settings::gust_frequency>("Gust Frequency", "Hz", {.minimum = 0.0, .maximum = 10.0, .step = 0.05, .application = recreate, .section = "Forces"}),
            spectra::sdk::parameter<"airDensity", &Settings::air_density>("Air Density", "kg/m³", {.minimum = 0.0, .maximum = 10.0, .step = 0.01, .application = recreate, .section = "Forces"}),
            spectra::sdk::parameter<"dragCoefficient", &Settings::drag_coefficient>("Drag Coefficient", {}, {.minimum = 0.0, .maximum = 4.0, .step = 0.01, .application = recreate, .section = "Forces"}),
            spectra::sdk::parameter<"substeps", &Settings::substeps>("Substeps", {}, {.minimum = 1.0, .maximum = 64.0, .step = 1.0, .application = recreate, .section = "Integration"}),
            spectra::sdk::mesh<"surface">(),
            spectra::sdk::mesh_field<"velocity", spectra::sdk::Float3>("Velocity", "m/s", {.name = "Velocity", .anchor = "surface", .kind = spectra::sdk::VisualizationKind::Vectors, .domain = spectra::sdk::MeshElementDomain::Vertex, .visible = false}),
            spectra::sdk::mesh_field<"force", spectra::sdk::Float3>("Last Substep Force", "N", {.name = "Last Substep Force", .anchor = "surface", .kind = spectra::sdk::VisualizationKind::Vectors, .domain = spectra::sdk::MeshElementDomain::Vertex, .visible = false}),
            spectra::sdk::mesh_field<"strain", float>("Maximum Stretch Strain", {}, {.name = "Maximum Stretch Strain", .anchor = "surface", .kind = spectra::sdk::VisualizationKind::Surface, .domain = spectra::sdk::MeshElementDomain::Vertex, .visible = false}),
            spectra::sdk::indexed_points<"pins">({.name = "Pinned Vertices", .anchor = "surface", .kind = spectra::sdk::VisualizationKind::Points, .domain = spectra::sdk::MeshElementDomain::Vertex, .visible = false}),
            spectra::sdk::indexed_segments<"stretch-constraints">({.name = "Stretch Constraints", .anchor = "surface", .kind = spectra::sdk::VisualizationKind::Segments, .domain = spectra::sdk::MeshElementDomain::Edge, .visible = false}),
            spectra::sdk::indexed_segments<"bend-constraints">({.name = "Bending Constraints", .anchor = "surface", .kind = spectra::sdk::VisualizationKind::Segments, .domain = spectra::sdk::MeshElementDomain::Edge, .visible = false}),
            spectra::sdk::metric<"step", std::uint64_t>("Step", {}, "Simulation"),
            spectra::sdk::metric<"time", double>("Physical Time", "s", "Simulation", true)
        );

        Module(Settings settings, const std::filesystem::path& assets, const spectra::sdk::SceneInputs& inputs);
        ~Module();

        Module(const Module&)            = delete;
        Module& operator=(const Module&) = delete;

        void setup(spectra::sdk::cuda::Setup& setup);
        void reset(std::uint64_t seed);
        void step(double seconds);
        void publish(spectra::sdk::cuda::Output& output, spectra::sdk::PresentationFrame presentation);

    private:
        std::unique_ptr<Simulation> simulation;

        [[nodiscard]] static Configuration configuration(const Settings& settings, const spectra::sdk::SceneInputs& inputs);
    };

} // namespace physica::examples::cloth_flag
