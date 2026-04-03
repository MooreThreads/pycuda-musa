#ifndef _ASDFDAFVVAFF_PYMUSA_HEADER_SEEN_TOOLS_HPP
#define _ASDFDAFVVAFF_PYMUSA_HEADER_SEEN_TOOLS_HPP




#include <musa.hpp>
#include <boost/python.hpp>
#include <numeric>
#include <numpy/arrayobject.h>




namespace pymusa
{
  inline
  npy_intp size_from_dims(size_t ndim, const npy_intp *dims)
  {
    if (ndim != 0)
      return std::accumulate(dims, dims+ndim, npy_intp(1), std::multiplies<npy_intp>());
    else
      return 1;
  }




  inline void run_python_gc()
  {
    namespace py = boost::python;

    py::object gc_mod(
        py::handle<>(
          PyImport_ImportModule("gc")));
    gc_mod.attr("collect")();
  }




  inline MUdeviceptr mem_alloc_gc(size_t bytes)
  {
    try
    {
      return pymusa::mem_alloc(bytes);
    }
    catch (pymusa::error &e)
    { 
      if (e.code() != MUSA_ERROR_OUT_OF_MEMORY)
        throw;
    }

    // If we get here, we got OUT_OF_MEMORY from CUDA.
    // We should run the Python GC to try and free up
    // some memory references.
    run_python_gc();

    // Now retry the allocation. If it fails again,
    // let it fail.
    return pymusa::mem_alloc(bytes);
  }
}





#endif
