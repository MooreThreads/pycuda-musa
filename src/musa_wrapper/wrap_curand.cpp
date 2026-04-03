#include <musa.hpp>
#include <murand.hpp>

#include "tools.hpp"
#include "wrap_helpers.hpp"

#if MUSAPP_MUSA_VERSION >= 3020
#include <murand.h>
#endif

using namespace pymusa;
using namespace pymusa::murandom;

void pymusa_expose_murand()
{
  using py::arg;
  using py::args;

#if MUSAPP_MUSA_VERSION >= 3020
  py::enum_<murandDirectionVectorSet_t>("direction_vector_set")
    .value("VECTOR_32", MURAND_DIRECTION_VECTORS_32_JOEKUO6)
#if MUSAPP_MUSA_VERSION >= 4000
    .value("SCRAMBLED_VECTOR_32", MURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6)
    .value("VECTOR_64", MURAND_DIRECTION_VECTORS_64_JOEKUO6)
    .value("SCRAMBLED_VECTOR_64", MURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6)
#endif
  ;
#endif

  py::def("get_murand_version", py_murand_version);

#if MUSAPP_MUSA_VERSION >= 3020
  py::def("_get_direction_vectors", py_murand_get_direction_vectors,
      (arg("set"), arg("dst"), arg("count")));
#endif

#if MUSAPP_MUSA_VERSION >= 4000
  py::def("_get_scramble_constants32", py_murand_get_scramble_constants32,
      (arg("dst"), arg("count")));
  py::def("_get_scramble_constants64", py_murand_get_scramble_constants64,
      (arg("dst"), arg("count")));
#endif
}
