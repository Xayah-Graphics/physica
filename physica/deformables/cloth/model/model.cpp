module;

#include <physica/cuda.h>

module physica.deformables.cloth.model;

import std;

namespace physica::deformables::cloth {
    Model::Model(ModelConfiguration next_configuration, const ::cuda::stream_ref source_stream) : configuration(std::move(next_configuration)), fields(source_stream), particle_count(configuration.rest_positions.size()) {}
}
