#ifndef _AFJDFJSDFSD_PYMUSA_HEADER_SEEN_CURAND_HPP
#define _AFJDFJSDFSD_PYMUSA_HEADER_SEEN_CURAND_HPP


#if MUSAPP_MUSA_VERSION >= 3020
  #include <murand.h>

  #ifdef MUSAPP_TRACE_MUSA
    #define MURAND_PRINT_ERROR_TRACE(NAME, CODE) \
      if (CODE != MURAND_STATUS_SUCCESS) \
        std::cerr << NAME << " failed with code " << CODE << std::endl;
  #else
    #define MURAND_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
  #endif

  #define MURAND_CALL_GUARDED(NAME, ARGLIST) \
    { \
      MUSAPP_PRINT_CALL_TRACE(#NAME); \
      murandStatus_t cu_status_code; \
      cu_status_code = NAME ARGLIST; \
      MURAND_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
      if (cu_status_code != MURAND_STATUS_SUCCESS) \
        throw pymusa::error(#NAME, MUSA_SUCCESS);\
    }
#else
  #define MURAND_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
  #define MURAND_CALL_GUARDED(NAME, ARGLIST) /*nothing*/
#endif


namespace pymusa { namespace murandom {

  py::tuple py_murand_version()
  {
    int version = 0;
#if MUSAPP_MUSA_VERSION >= 3020
    murandGetVersion(&version);
#endif
    return py::make_tuple(
        version / 1000,
        (version % 1000)/10,
        version % 10);
  }

#if MUSAPP_MUSA_VERSION >= 3020
  void py_murand_get_direction_vectors(
      murandDirectionVectorSet_t set, py::object dst, int count)
  {
    int n = 0;

    py_buffer_wrapper buf_wrapper;
    buf_wrapper.get(dst.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    void *buf = buf_wrapper.m_buf.buf;

    if (MURAND_DIRECTION_VECTORS_32_JOEKUO6 == set
#if MUSAPP_MUSA_VERSION >= 4000
      || MURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 == set
#endif
    ) {
      murandDirectionVectors32_t *vectors;
      MURAND_CALL_GUARDED(murandGetDirectionVectors32, (&vectors, set));
      while (count > 0) {
        int size = ((count > 20000) ? 20000 : count)*sizeof(murandDirectionVectors32_t);
        memcpy((unsigned int *)buf+n*20000*sizeof(murandDirectionVectors32_t)/sizeof(unsigned int), vectors, size);
	count -= size/sizeof(murandDirectionVectors32_t);
        n++;
      }
    }
#if MUSAPP_MUSA_VERSION >= 4000
    if (MURAND_DIRECTION_VECTORS_64_JOEKUO6 == set
      || MURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 == set) {
      murandDirectionVectors64_t *vectors;
      MURAND_CALL_GUARDED(murandGetDirectionVectors64, (&vectors, set));
      while (count > 0) {
        int size = ((count > 20000) ? 20000 : count)*sizeof(murandDirectionVectors64_t);
        memcpy((unsigned long long *)buf+n*20000*sizeof(murandDirectionVectors64_t)/sizeof(unsigned long long), vectors, size);
	count -= size/sizeof(murandDirectionVectors64_t);
        n++;
      }
    }
#endif
  }
#endif

#if MUSAPP_MUSA_VERSION >= 4000
  void py_murand_get_scramble_constants32(py::object dst, int count)
  {
    int n = 0;

    py_buffer_wrapper buf_wrapper;
    buf_wrapper.get(dst.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    void *buf = buf_wrapper.m_buf.buf;
    PYMUSA_BUFFER_SIZE_T len = buf_wrapper.m_buf.len;

    unsigned int *vectors;
    MURAND_CALL_GUARDED(murandGetScrambleConstants32, (&vectors));
// Documentation does not mention number of dimensions
// Assuming the same as in getDirectionVectors*
    while (count > 0) {
      int size = ((count > 20000) ? 20000 : count)*sizeof(unsigned int);
      memcpy((unsigned int *)buf+n*20000, vectors, size);
      count -= size/sizeof(unsigned int);
      n++;
    }
  }

  void py_murand_get_scramble_constants64(py::object dst, int count)
  {
    int n = 0;

    py_buffer_wrapper buf_wrapper;
    buf_wrapper.get(dst.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE);

    void *buf = buf_wrapper.m_buf.buf;
    PYMUSA_BUFFER_SIZE_T len = buf_wrapper.m_buf.len;

    unsigned long long *vectors;
    MURAND_CALL_GUARDED(murandGetScrambleConstants64, (&vectors));
// Documentation does not mention number of dimensions
// Assuming the same as in getDirectionVectors*
    while (count > 0) {
      int size = ((count > 20000) ? 20000 : count)*sizeof(unsigned long long);
      memcpy((unsigned long long *)buf+n*20000, vectors, size);
      count -= size/sizeof(unsigned long long);
      n++;
    }
  }
#endif
} }

#endif

