// A C++ wrapper for CUDA




#ifndef _AFJDFJSDFSD_PYMUSA_HEADER_SEEN_CUDA_HPP
#define _AFJDFJSDFSD_PYMUSA_HEADER_SEEN_CUDA_HPP




// {{{ includes, configuration

#include <musa.h>

#ifdef MUSAPP_PRETEND_MUSA_VERSION
#define MUSAPP_MUSA_VERSION MUSAPP_PRETEND_MUSA_VERSION
#else
#define MUSAPP_MUSA_VERSION MUSA_VERSION
#endif

#if MUSAPP_MUSA_VERSION >= 4000
#include <musaProfiler.h>
#endif

#ifndef _MSC_VER
#include <stdint.h>
#endif
#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include <utility>
#include <stack>
#include <iostream>
#include <vector>
#include <boost/python.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/tss.hpp>
#include <boost/version.hpp>

#if (BOOST_VERSION/100) < 1035
#warning *****************************************************************
#warning **** Your version of Boost C++ is likely too old for PyCUDA. ****
#warning *****************************************************************
#endif

// MAYBE? muMemcpy, muPointerGetAttribute
// TODO: muCtxSetCurrent, muCtxGetCurrent
// (use once the old, deprecated functions have been removed from CUDA)




// #define MUSAPP_TRACE_MUSA
#define MUSAPP_POST_30_BETA

#ifdef MUSAPP_PRETEND_MUSA_VERSION
#define MUSAPP_MUSA_VERSION MUSAPP_PRETEND_MUSA_VERSION
#else
#define MUSAPP_MUSA_VERSION MUSA_VERSION
#endif




#if (PY_VERSION_HEX < 0x02060000)
  #error PyCUDA does not support Python 2 versions earlier than 2.6.
#endif
#if (PY_VERSION_HEX >= 0x03000000) && (PY_VERSION_HEX < 0x03030000)
  #error PyCUDA does not support Python 3 versions earlier than 3.3.
#endif

typedef Py_ssize_t PYMUSA_BUFFER_SIZE_T;

// }}}


#define PYMUSA_PARSE_STREAM_PY \
    MUstream s_handle; \
    if (stream_py.ptr() != Py_None) \
    { \
      const stream &s = py::extract<const stream &>(stream_py); \
      s_handle = s.handle(); \
    } \
    else \
      s_handle = 0;



// {{{ tracing and error guards

#ifdef MUSAPP_TRACE_MUSA
  #define MUSAPP_PRINT_CALL_TRACE(NAME) \
    std::cerr << NAME << std::endl;
  #define MUSAPP_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) \
    std::cerr << NAME << " (" << EXTRA_INFO << ')' << std::endl;
  #define CUDAPP_PRINT_ERROR_TRACE(NAME, CODE) \
    if (CODE != MUSA_SUCCESS) \
      std::cerr << NAME << " failed with code " << CODE << std::endl;
#else
  #define MUSAPP_PRINT_CALL_TRACE(NAME) /*nothing*/
  #define MUSAPP_PRINT_CALL_TRACE_INFO(NAME, EXTRA_INFO) /*nothing*/
  #define CUDAPP_PRINT_ERROR_TRACE(NAME, CODE) /*nothing*/
#endif

#define MUSAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
  { \
    MUSAPP_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
    MUresult cu_status_code; \
    Py_BEGIN_ALLOW_THREADS \
      cu_status_code = NAME ARGLIST; \
    Py_END_ALLOW_THREADS \
    if (cu_status_code != MUSA_SUCCESS) \
      throw pymusa::error(#NAME, cu_status_code);\
  }

#define MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(NAME, ARGLIST, TRACE_INFO) \
  { \
    MUSAPP_PRINT_CALL_TRACE_INFO(#NAME, TRACE_INFO); \
    MUresult cu_status_code; \
    cu_status_code = NAME ARGLIST; \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != MUSA_SUCCESS) \
      throw pymusa::error(#NAME, cu_status_code);\
  }

#define MUSAPP_CALL_GUARDED_THREADED(NAME, ARGLIST) \
  { \
    MUSAPP_PRINT_CALL_TRACE(#NAME); \
    MUresult cu_status_code; \
    Py_BEGIN_ALLOW_THREADS \
      cu_status_code = NAME ARGLIST; \
    Py_END_ALLOW_THREADS \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != MUSA_SUCCESS) \
      throw pymusa::error(#NAME, cu_status_code);\
  }

#define MUSAPP_CALL_GUARDED(NAME, ARGLIST) \
  { \
    MUSAPP_PRINT_CALL_TRACE(#NAME); \
    MUresult cu_status_code; \
    cu_status_code = NAME ARGLIST; \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != MUSA_SUCCESS) \
      throw pymusa::error(#NAME, cu_status_code);\
  }
#define MUSAPP_CALL_GUARDED_CLEANUP(NAME, ARGLIST) \
  { \
    MUSAPP_PRINT_CALL_TRACE(#NAME); \
    MUresult cu_status_code; \
    cu_status_code = NAME ARGLIST; \
    CUDAPP_PRINT_ERROR_TRACE(#NAME, cu_status_code); \
    if (cu_status_code != MUSA_SUCCESS) \
      std::cerr \
        << "PyCUDA WARNING: a clean-up operation failed (dead context maybe?)" \
        << std::endl \
        << pymusa::error::make_message(#NAME, cu_status_code) \
        << std::endl; \
  }
#define CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(TYPE) \
  catch (pymusa::cannot_activate_out_of_thread_context &e) \
  { \
    PyErr_Warn( \
        PyExc_UserWarning, #TYPE " in out-of-thread context could not be cleaned up"); \
  } \
  catch (pymusa::cannot_activate_dead_context &e) \
  { \
    /* PyErr_Warn( \
        PyExc_UserWarning, #TYPE " in dead context was implicitly cleaned up");*/ \
  }
  // In all likelihood, this TYPE's managing thread has exited, and
  // therefore its context has already been deleted. No need to harp
  // on the fact that we still thought there was cleanup to do.

// }}}




namespace pymusa
{
  namespace py = boost::python;

  typedef
#if MUSAPP_MUSA_VERSION >= 3020
        size_t
#else
        unsigned int
#endif
        pymusa_size_t;

  typedef
#if defined(_WIN32) && defined(_WIN64)
    long long
#else
    long
#endif
    hash_type;



  // {{{ error reporting
  class error : public std::runtime_error
  {
    private:
      const char *m_routine;
      MUresult m_code;

    public:
      static std::string make_message(const char *rout, MUresult c, const char *msg=0)
      {
        std::string result = rout;
        result += " failed: ";
        result += curesult_to_str(c);
        if (msg)
        {
          result += " - ";
          result += msg;
        }
        return result;
      }

      error(const char *rout, MUresult c, const char *msg=0)
        : std::runtime_error(make_message(rout, c, msg)),
        m_routine(rout), m_code(c)
      { }

      const char *routine() const
      {
        return m_routine;
      }

      MUresult code() const
      {
        return m_code;
      }

      bool is_out_of_memory() const
      {
        return code() == MUSA_ERROR_OUT_OF_MEMORY;
      }

      static const char *curesult_to_str(MUresult e)
      {
#if MUSAPP_MUSA_VERSION >= 6000
        const char* errstr;
        muGetErrorString(e, &errstr);
        return errstr;
#else
        switch (e)
        {
          case MUSA_SUCCESS: return "success";
          case MUSA_ERROR_INVALID_VALUE: return "imtalid value";
          case MUSA_ERROR_OUT_OF_MEMORY: return "out of memory";
          case MUSA_ERROR_NOT_INITIALIZED: return "not initialized";
#if MUSAPP_MUSA_VERSION >= 2000
          case MUSA_ERROR_DEINITIALIZED: return "deinitialized";
#endif
#if MUSAPP_MUSA_VERSION >= 4000
          case MUSA_ERROR_PROFILER_DISABLED: return "profiler disabled";
          case MUSA_ERROR_PROFILER_NOT_INITIALIZED: return "profiler not initialized";
          case MUSA_ERROR_PROFILER_ALREADY_STARTED: return "profiler already started";
          case MUSA_ERROR_PROFILER_ALREADY_STOPPED: return "profiler already stopped";
#endif

          case MUSA_ERROR_NO_DEVICE: return "no device";
          case MUSA_ERROR_INVALID_DEVICE: return "imtalid device";

          case MUSA_ERROR_INVALID_IMAGE: return "imtalid image";
          case MUSA_ERROR_INVALID_CONTEXT: return "imtalid context";
          case MUSA_ERROR_CONTEXT_ALREADY_CURRENT: return "context already current";
          case MUSA_ERROR_MAP_FAILED: return "map failed";
          case MUSA_ERROR_UNMAP_FAILED: return "unmap failed";
          case MUSA_ERROR_ARRAY_IS_MAPPED: return "array is mapped";
          case MUSA_ERROR_ALREADY_MAPPED: return "already mapped";
          case MUSA_ERROR_NO_BINARY_FOR_GPU: return "no binary for gpu";
          case MUSA_ERROR_ALREADY_ACQUIRED: return "already acquired";
          case MUSA_ERROR_NOT_MAPPED: return "not mapped";
#if MUSAPP_MUSA_VERSION >= 3000
          case MUSA_ERROR_NOT_MAPPED_AS_ARRAY: return "not mapped as array";
          case MUSA_ERROR_NOT_MAPPED_AS_POINTER: return "not mapped as pointer";
#ifdef MUSAPP_POST_30_BETA
          case MUSA_ERROR_ECC_UNCORRECTABLE: return "ECC uncorrectable";
#endif
#endif
#if MUSAPP_MUSA_VERSION >= 3010
          case MUSA_ERROR_UNSUPPORTED_LIMIT: return "unsupported limit";
#endif
#if MUSAPP_MUSA_VERSION >= 4000
          case MUSA_ERROR_CONTEXT_ALREADY_IN_USE: return "context already in use";
#endif

          case MUSA_ERROR_INVALID_SOURCE: return "imtalid source";
          case MUSA_ERROR_FILE_NOT_FOUND: return "file not found";
#if MUSAPP_MUSA_VERSION >= 3010
          case MUSA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "shared object symbol not found";
          case MUSA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "shared object init failed";
#endif

          case MUSA_ERROR_INVALID_HANDLE: return "imtalid handle";

          case MUSA_ERROR_NOT_FOUND: return "not found";

          case MUSA_ERROR_NOT_READY: return "not ready";

          case MUSA_ERROR_LAUNCH_FAILED: return "launch failed";
          case MUSA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "launch out of resources";
          case MUSA_ERROR_LAUNCH_TIMEOUT: return "launch timeout";
          case MUSA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "launch incompatible texturing";

#if MUSAPP_MUSA_VERSION >= 4000
          case MUSA_ERROR_PEER_ACCESS_ALREADY_ENABLED: return "peer access already enabled";
          case MUSA_ERROR_PEER_ACCESS_NOT_ENABLED: return "peer access not enabled";
          case MUSA_ERROR_PRIMARY_CONTEXT_ACTIVE: return "primary context active";
          case MUSA_ERROR_CONTEXT_IS_DESTROYED: return "context is destroyed";
#endif

#if (MUSAPP_MUSA_VERSION >= 3000) && (MUSAPP_MUSA_VERSION < 3020)
          case CUDA_ERROR_POINTER_IS_64BIT:
             return "attempted to retrieve 64-bit pointer via 32-bit api function";
          case CUDA_ERROR_SIZE_IS_64BIT:
             return "attempted to retrieve 64-bit size via 32-bit api function";
#endif

#if MUSAPP_MUSA_VERSION >= 4010
          case MUSA_ERROR_ASSERT:
             return "device-side assert triggered";
          case MUSA_ERROR_TOO_MANY_PEERS:
             return "too many peers";
          case MUSA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
             return "host memory already registered";
          case MUSA_ERROR_HOST_MEMORY_NOT_REGISTERED:
             return "host memory not registered";
#endif

#if MUSAPP_MUSA_VERSION >= 5000
          case MUSA_ERROR_NOT_SUPPORTED:
             return "operation not supported on current system or device";
#endif

          case MUSA_ERROR_UNKNOWN: return "unknown";

          default: return "imtalid/unknown error code";
        }
#endif
      }
  };

  struct cannot_activate_out_of_thread_context : public std::logic_error
  {
    cannot_activate_out_of_thread_context(std::string const &w)
      : std::logic_error(w)
    { }
  };

  struct cannot_activate_dead_context : public std::logic_error
  {
    cannot_activate_dead_context(std::string const &w)
      : std::logic_error(w)
    { }
  };

  // }}}

  // {{{ buffer interface helper

  class py_buffer_wrapper : public boost::noncopyable
  {
    private:
      bool m_initialized;

    public:
      Py_buffer m_buf;

      py_buffer_wrapper()
        : m_initialized(false)
      {}

      void get(PyObject *obj, int flags)
      {
        if (PyObject_GetBuffer(obj, &m_buf, flags))
          throw py::error_already_set();

        m_initialized = true;
      }

      virtual ~py_buffer_wrapper()
      {
        if (m_initialized)
          PyBuffer_Release(&m_buf);
      }
  };

  // }}}


  // {{{ version query ------------------------------------------------------------
#if MUSAPP_MUSA_VERSION >= 2020
  inline int get_driver_version()
  {
    int result;
    MUSAPP_CALL_GUARDED(muDriverGetVersion, (&result));
    return result;
  }
#endif
  // }}}

  // {{{ device
  class context;
  class primary_context;

  class device
  {
    private:
      MUdevice m_device;

    public:
      device(MUdevice dev)
        : m_device(dev)
      { }

      static int count()
      {
        int result;
        MUSAPP_CALL_GUARDED(muDeviceGetCount, (&result));
        return result;
      }

      std::string name()
      {
        char buffer[1024];
        MUSAPP_CALL_GUARDED(muDeviceGetName, (buffer, sizeof(buffer), m_device));
        return buffer;
      }

#if MUSAPP_MUSA_VERSION >= 4010
      std::string pci_bus_id()
      {
        char buffer[1024];
        MUSAPP_CALL_GUARDED(muDeviceGetPCIBusId, (buffer, sizeof(buffer), m_device));
        return buffer;
      }
#endif

      py::tuple compute_capability()
      {
        int major, minor;
        MUSAPP_CALL_GUARDED(muDeviceComputeCapability, (&major, &minor, m_device));
        return py::make_tuple(major, minor);
      }

      pymusa_size_t total_memory()
      {
        pymusa_size_t bytes;

        MUSAPP_CALL_GUARDED(muDeviceTotalMem, (&bytes, m_device));
        return bytes;
      }

      int get_attribute(MUdevice_attribute attr) const
      {
        int result;
        MUSAPP_CALL_GUARDED(muDeviceGetAttribute, (&result, attr, m_device));
        return result;
      }

      bool operator==(const device &other) const
      {
        return m_device == other.m_device;
      }

      bool operator!=(const device &other) const
      {
        return m_device != other.m_device;
      }

      hash_type hash() const
      {
        return m_device;
      }

      boost::shared_ptr<context> make_context(unsigned int flags);
#if MUSAPP_MUSA_VERSION >= 7000
      boost::shared_ptr<context> retain_primary_context();
#endif

      MUdevice handle() const
      { return m_device; }

#if MUSAPP_MUSA_VERSION >= 4000
      bool can_access_peer(device const &other)
      {
        int result;
        MUSAPP_CALL_GUARDED(muDeviceCanAccessPeer, (&result, handle(), other.handle()));
        return result;
      }
#endif

  };

  inline
  void init(unsigned int flags)
  {
    MUSAPP_CALL_GUARDED_THREADED(muInit, (flags));
  }

  inline
  device *make_device(int ordinal)
  {
    MUdevice result;
    MUSAPP_CALL_GUARDED(muDeviceGet, (&result, ordinal));
    return new device(result);
  }

#if MUSAPP_MUSA_VERSION >= 4010
  inline
  device *make_device_from_pci_bus_id(std::string const pci_bus_id)
  {
    MUdevice result;
    MUSAPP_CALL_GUARDED(muDeviceGetByPCIBusId, (&result,
          const_cast<char *>(pci_bus_id.c_str())));
    return new device(result);
  }
#endif

  // }}}

  // {{{ context
  /* A word on context management: We don't let CUDA's context stack get more
   * than one deep. CUDA only supports pushing floating contexts. We may wish
   * to push contexts that are already active at a deeper stack level, so we
   * maintain all contexts floating other than the top one.
   */

  // for friend decl
  namespace gl {
    boost::shared_ptr<context>
        make_gl_context(device const &dev, unsigned int flags);
  }

  class context_stack;
  extern boost::thread_specific_ptr<context_stack> context_stack_ptr;

  class context_stack
  {
      /* This wrapper is necessary because we need to pop the contents
       * off the stack before we destroy each of the contexts. This, in turn,
       * is because the contexts need to be able to access the stack in order
       * to be destroyed.
       */
    private:
      typedef std::stack<boost::shared_ptr<context> > stack_t;
      typedef stack_t::value_type value_type;;
      stack_t m_stack;

    public:
      ~context_stack();

      bool empty() const
      { return m_stack.empty(); }

      value_type &top()
      { return m_stack.top(); }

      void pop()
      {
        if (m_stack.empty())
        {
          throw error("m_stack::pop", MUSA_ERROR_INVALID_CONTEXT,
              "cannot pop context from empty stack");
        }
        m_stack.pop();
      }

      void push(value_type v)
      { m_stack.push(v); }

      static context_stack &get()
      {
        if (context_stack_ptr.get() == 0)
          context_stack_ptr.reset(new context_stack);

        return *context_stack_ptr;
      }
  };

  class context : boost::noncopyable
  {
    protected:
      MUcontext m_context;
      bool m_valid;
      unsigned m_use_count;
      boost::thread::id m_thread;

    public:
      context(MUcontext ctx)
        : m_context(ctx), m_valid(true), m_use_count(1),
        m_thread(boost::this_thread::get_id())
      { }

      virtual ~context()
      {
        if (m_valid)
        {
          /* It's possible that we get here with a non-zero m_use_count. Since the context
           * stack holds shared_ptrs, this must mean that the context stack itself is getting
           * destroyed, which means it's ok for this context to sign off, too.
           */
          detach();
        }
      }

      MUcontext handle() const
      { return m_context; }

      intptr_t handle_int() const
      { return (intptr_t) m_context; }

      bool operator==(const context &other) const
      {
        return m_context == other.m_context;
      }

      bool operator!=(const context &other) const
      {
        return m_context != other.m_context;
      }

      hash_type hash() const
      {
        return hash_type(m_context) ^ hash_type(this);
      }

      boost::thread::id thread_id() const
      { return m_thread; }

      bool is_valid() const
      {
        return m_valid;
      }

      static boost::shared_ptr<context> attach(unsigned int flags)
      {
        MUcontext current;
        MUSAPP_CALL_GUARDED(muCtxAttach, (&current, flags));
        boost::shared_ptr<context> result(new context(current));
        context_stack::get().push(result);
        return result;
      }

    protected:
      virtual void detach_internal()
      {
        MUSAPP_CALL_GUARDED_CLEANUP(muCtxDetach, (m_context));
      }

    public:
      virtual void detach()
      {
        if (m_valid)
        {
          bool active_before_destruction = current_context().get() == this;
          if (active_before_destruction)
          {
            detach_internal();
          }
          else
          {
            if (m_thread == boost::this_thread::get_id())
            {
              MUSAPP_CALL_GUARDED_CLEANUP(muCtxPushCurrent, (m_context));
              detach_internal();
              /* pop is implicit in detach */
            }
            else
            {
              // In all likelihood, this context's managing thread has exited, and
              // therefore this context has already been deleted. No need to harp
              // on the fact that we still thought there was cleanup to do.

              // std::cerr << "PyCUDA WARNING: leaked out-of-thread context " << std::endl;
            }
          }

          m_valid = false;

          if (active_before_destruction)
          {
            boost::shared_ptr<context> new_active = current_context(this);
            if (new_active.get())
            {
              MUSAPP_CALL_GUARDED(muCtxPushCurrent, (new_active->m_context));
            }
          }
        }
        else
          throw error("context::detach", MUSA_ERROR_INVALID_CONTEXT,
              "cannot detach from imtalid context");
      }

      static device get_device()
      {
        MUdevice dev;
        MUSAPP_CALL_GUARDED(muCtxGetDevice, (&dev));
        return device(dev);
      }

#if MUSAPP_MUSA_VERSION >= 2000

      static void prepare_context_switch()
      {
        if (!context_stack::get().empty())
        {
          MUcontext popped;
          MUSAPP_CALL_GUARDED(muCtxPopCurrent, (&popped));
        }
      }

      static void pop()
      {
        prepare_context_switch();
        context_stack &ctx_stack = context_stack::get();

        if (ctx_stack.empty())
        {
          throw error("context::pop", MUSA_ERROR_INVALID_CONTEXT,
              "cannot pop non-current context");
        }

        boost::shared_ptr<context> current = current_context();
        if (current)
          --current->m_use_count;

        ctx_stack.pop();

        current = current_context();
        if (current)
          MUSAPP_CALL_GUARDED(muCtxPushCurrent, (current_context()->m_context));
      }
#else
      static void prepare_context_switch() { }
#endif

      static void synchronize()
      { MUSAPP_CALL_GUARDED_THREADED(muCtxSynchronize, ()); }

      static boost::shared_ptr<context> current_context(context *except=0)
      {
        while (true)
        {
          if (context_stack::get().empty())
            return boost::shared_ptr<context>();

          boost::shared_ptr<context> result(context_stack::get().top());
          if (result.get() != except
              && result->is_valid())
          {
            // good, weak pointer didn't expire
            return result;
          }

          // context imtalid, pop it and try again.
          context_stack::get().pop();
        }
      }

#if MUSAPP_MUSA_VERSION >= 3010
      static void set_limit(MUlimit limit, size_t value)
      {
        MUSAPP_CALL_GUARDED(muCtxSetLimit, (limit, value));
      }

      static size_t get_limit(MUlimit limit)
      {
        size_t value;
        MUSAPP_CALL_GUARDED(muCtxGetLimit, (&value, limit));
        return value;
      }
#endif

#if MUSAPP_MUSA_VERSION >= 3020
      static MUfunc_cache get_cache_config()
      {
        MUfunc_cache value;
        MUSAPP_CALL_GUARDED(muCtxGetCacheConfig, (&value));
        return value;
      }

      static void set_cache_config(MUfunc_cache cc)
      {
        MUSAPP_CALL_GUARDED(muCtxSetCacheConfig, (cc));
      }

      unsigned int get_api_version()
      {
        unsigned int value;
        MUSAPP_CALL_GUARDED(muCtxGetApiVersion, (m_context, &value));
        return value;
      }
#endif

#if MUSAPP_MUSA_VERSION >= 4000
      static void enable_peer_access(context const &peer, unsigned int flags)
      {
        MUSAPP_CALL_GUARDED(muCtxEnablePeerAccess, (peer.handle(), flags));
      }

      static void disable_peer_access(context const &peer)
      {
        MUSAPP_CALL_GUARDED(muCtxDisablePeerAccess, (peer.handle()));
      }
#endif

#if MUSAPP_MUSA_VERSION >= 4020
      static MUsharedconfig get_shared_config()
      {
        MUsharedconfig config;
        MUSAPP_CALL_GUARDED(muCtxGetSharedMemConfig, (&config));
        return config;
      }

      static void set_shared_config(MUsharedconfig config)
      {
        MUSAPP_CALL_GUARDED(muCtxSetSharedMemConfig, (config));
      }
#endif

      friend class device;
      friend void context_push(boost::shared_ptr<context> ctx);
      friend boost::shared_ptr<context>
          gl::make_gl_context(device const &dev, unsigned int flags);
      friend class primary_context;
  };

  class primary_context : public context
  {
   protected:
      MUdevice m_device;

    public:
      primary_context(MUcontext ctx, MUdevice dev)
        : context (ctx), m_device(dev)
      { }

    protected:
      virtual void detach_internal()
      {
        // Primary context comes from retainPrimaryContext.
        MUSAPP_CALL_GUARDED_CLEANUP(muDevicePrimaryCtxRelease, (m_device));
      }
  };

  inline
  boost::shared_ptr<context> device::make_context(unsigned int flags)
  {
    context::prepare_context_switch();

    MUcontext ctx;

#if MUSAPP_MUSA_VERSION >= 13000
    MUSAPP_CALL_GUARDED_THREADED(muCtxCreate, (&ctx, nullptr, flags, m_device));
#else
    MUSAPP_CALL_GUARDED_THREADED(muCtxCreate, (&ctx, flags, m_device));
#endif

    boost::shared_ptr<context> result(new context(ctx));
    context_stack::get().push(result);
    return result;
  }


#if MUSAPP_MUSA_VERSION >= 7000
  inline boost::shared_ptr<context> device::retain_primary_context()
  {
    MUcontext ctx;
    MUSAPP_CALL_GUARDED(muDevicePrimaryCtxRetain, (&ctx, m_device));
    boost::shared_ptr<context> result(new primary_context(ctx, m_device));
    return result;
  }
#endif


#if MUSAPP_MUSA_VERSION >= 2000
  inline
  void context_push(boost::shared_ptr<context> ctx)
  {
    context::prepare_context_switch();

    MUSAPP_CALL_GUARDED(muCtxPushCurrent, (ctx->m_context));
    context_stack::get().push(ctx);
    ++ctx->m_use_count;
  }
#endif




  inline context_stack::~context_stack()
  {
    if (!m_stack.empty())
    {
      std::cerr
        << "-------------------------------------------------------------------" << std::endl
        << "PyCUDA ERROR: The context stack was not empty upon module cleanup." << std::endl
        << "-------------------------------------------------------------------" << std::endl
        << "A context was still active when the context stack was being" << std::endl
        << "cleaned up. At this point in our execution, CUDA may already" << std::endl
        << "have been deinitialized, so there is no way we can finish" << std::endl
        << "cleanly. The program will be aborted now." << std::endl
        << "Use Context.pop() to avoid this problem." << std::endl
        << "-------------------------------------------------------------------" << std::endl;
      abort();
    }
  }




  class explicit_context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      void acquire_context()
      {
        m_ward_context = context::current_context();
        if (m_ward_context.get() == 0)
          throw error("explicit_context_dependent",
              MUSA_ERROR_INVALID_CONTEXT,
              "no currently active context?");
      }

      void release_context()
      {
        m_ward_context.reset();
      }

      boost::shared_ptr<context> get_context()
      {
        return m_ward_context;
      }
  };

  class context_dependent : public explicit_context_dependent
  {
    private:
      boost::shared_ptr<context> m_ward_context;

    public:
      context_dependent()
      { acquire_context(); }
  };


  class scoped_context_activation
  {
    private:
      boost::shared_ptr<context> m_context;
      bool m_did_switch;

    public:
      scoped_context_activation(boost::shared_ptr<context> ctx)
        : m_context(ctx)
      {
        if (!m_context->is_valid())
          throw pymusa::cannot_activate_dead_context(
              "cannot activate dead context");

        m_did_switch = context::current_context() != m_context;
        if (m_did_switch)
        {
          if (boost::this_thread::get_id() != m_context->thread_id())
            throw pymusa::cannot_activate_out_of_thread_context(
                "cannot activate out-of-thread context");
#if MUSAPP_MUSA_VERSION >= 2000
          context_push(m_context);
#else
          throw pymusa::error("scoped_context_activation", MUSA_ERROR_INVALID_CONTEXT,
              "not available in CUDA < 2.0");
#endif
        }
      }

      ~scoped_context_activation()
      {
#if MUSAPP_MUSA_VERSION >= 2000
        if (m_did_switch)
          m_context->pop();
#endif
      }

  };

  // }}}

  // {{{ stream
  class event;

  class stream : public boost::noncopyable, public context_dependent
  {
    private:
      MUstream m_stream;

    public:
      stream(unsigned int flags=0)
      { MUSAPP_CALL_GUARDED(muStreamCreate, (&m_stream, flags)); }

      ~stream()
      {
        try
        {
          scoped_context_activation ca(get_context());
          MUSAPP_CALL_GUARDED_CLEANUP(muStreamDestroy, (m_stream));
        }
        CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(stream);
      }

      void synchronize()
      { MUSAPP_CALL_GUARDED_THREADED(muStreamSynchronize, (m_stream)); }

      MUstream handle() const
      { return m_stream; }

      intptr_t handle_int() const
      { return (intptr_t) m_stream; }

#if MUSAPP_MUSA_VERSION >= 3020
      void wait_for_event(const event &evt);
#endif

      bool is_done() const
      {
        MUSAPP_PRINT_CALL_TRACE("muStreamQuery");
        MUresult result = muStreamQuery(m_stream);
        switch (result)
        {
          case MUSA_SUCCESS:
            return true;
          case MUSA_ERROR_NOT_READY:
            return false;
          default:
            CUDAPP_PRINT_ERROR_TRACE("muStreamQuery", result);
            throw error("muStreamQuery", result);
        }
      }
  };

  // }}}

  // {{{ array
  class array : public boost::noncopyable, public context_dependent
  {
    private:
      MUarray m_array;
      bool m_managed;

    public:
      array(const MUSA_ARRAY_DESCRIPTOR &descr)
        : m_managed(true)
      { MUSAPP_CALL_GUARDED(muArrayCreate, (&m_array, &descr)); }

#if MUSAPP_MUSA_VERSION >= 2000
      array(const MUSA_ARRAY3D_DESCRIPTOR &descr)
        : m_managed(true)
      { MUSAPP_CALL_GUARDED(muArray3DCreate, (&m_array, &descr)); }
#endif

      array(MUarray ary, bool managed)
        : m_array(ary), m_managed(managed)
      { }

      ~array()
      { free(); }

      void free()
      {
        if (m_managed)
        {
          try
          {
            scoped_context_activation ca(get_context());
            MUSAPP_CALL_GUARDED_CLEANUP(muArrayDestroy, (m_array));
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(array);

          m_managed = false;
          release_context();
        }
      }

      MUSA_ARRAY_DESCRIPTOR get_descriptor()
      {
        MUSA_ARRAY_DESCRIPTOR result;
        MUSAPP_CALL_GUARDED(muArrayGetDescriptor, (&result, m_array));
        return result;
      }

#if MUSAPP_MUSA_VERSION >= 2000
      MUSA_ARRAY3D_DESCRIPTOR get_descriptor_3d()
      {
        MUSA_ARRAY3D_DESCRIPTOR result;
        MUSAPP_CALL_GUARDED(muArray3DGetDescriptor, (&result, m_array));
        return result;
      }
#endif

      MUarray handle() const
      { return m_array; }

    intptr_t handle_int() const
    { return  (intptr_t) m_array; }
  };

  // }}}

  // {{{ texture reference
  class module;

  class texture_reference : public  boost::noncopyable
  {
    private:
      MUtexref m_texref;
      bool m_managed;

      // life support for array and module
      boost::shared_ptr<array> m_array;
      boost::shared_ptr<module> m_module;

    public:
      texture_reference()
        : m_managed(true)
      { MUSAPP_CALL_GUARDED(muTexRefCreate, (&m_texref)); }

      texture_reference(MUtexref tr, bool managed)
        : m_texref(tr), m_managed(managed)
      { }

      ~texture_reference()
      {
        if (m_managed)
        {
          MUSAPP_CALL_GUARDED_CLEANUP(muTexRefDestroy, (m_texref));
        }
      }

      void set_module(boost::shared_ptr<module> mod)
      { m_module = mod; }

      MUtexref handle() const
      { return m_texref; }

      void set_array(boost::shared_ptr<array> ary)
      {
        MUSAPP_CALL_GUARDED(muTexRefSetArray, (m_texref,
            ary->handle(), MU_TRSA_OVERRIDE_FORMAT));
        m_array = ary;
      }

      pymusa_size_t set_address(MUdeviceptr dptr, unsigned int bytes, bool allow_offset=false)
      {
        pymusa_size_t byte_offset;
        MUSAPP_CALL_GUARDED(muTexRefSetAddress, (&byte_offset,
              m_texref, dptr, bytes));

        if (!allow_offset && byte_offset != 0)
          throw pymusa::error("texture_reference::set_address", MUSA_ERROR_INVALID_VALUE,
              "texture binding resulted in offset, but allow_offset was false");

        m_array.reset();
        return byte_offset;
      }

#if MUSAPP_MUSA_VERSION >= 2020
      void set_address_2d(MUdeviceptr dptr,
          const MUSA_ARRAY_DESCRIPTOR &descr, unsigned int pitch)
      {
        MUSAPP_CALL_GUARDED(muTexRefSetAddress2D, (m_texref, &descr, dptr, pitch));
      }
#endif

      void set_format(MUarray_format fmt, int num_packed_components)
      { MUSAPP_CALL_GUARDED(muTexRefSetFormat, (m_texref, fmt, num_packed_components)); }

      void set_address_mode(int dim, MUaddress_mode am)
      { MUSAPP_CALL_GUARDED(muTexRefSetAddressMode, (m_texref, dim, am)); }
      void set_filter_mode(MUfilter_mode fm)
      { MUSAPP_CALL_GUARDED(muTexRefSetFilterMode, (m_texref, fm)); }

      void set_flags(unsigned int flags)
      { MUSAPP_CALL_GUARDED(muTexRefSetFlags, (m_texref, flags)); }

      MUdeviceptr get_address()
      {
        MUdeviceptr result;
        MUSAPP_CALL_GUARDED(muTexRefGetAddress, (&result, m_texref));
        return result;
      }
      array *get_array()
      {
        MUarray result;
        MUSAPP_CALL_GUARDED(muTexRefGetArray, (&result, m_texref));
        return new array(result, false);
      }
      MUaddress_mode get_address_mode(int dim)
      {
        MUaddress_mode result;
        MUSAPP_CALL_GUARDED(muTexRefGetAddressMode, (&result, m_texref, dim));
        return result;
      }
      MUfilter_mode get_filter_mode()
      {
        MUfilter_mode result;
        MUSAPP_CALL_GUARDED(muTexRefGetFilterMode, (&result, m_texref));
        return result;
      }

#if MUSAPP_MUSA_VERSION >= 2000
      py::tuple get_format()
      {
        MUarray_format fmt;
        int num_channels;
        MUSAPP_CALL_GUARDED(muTexRefGetFormat, (&fmt, &num_channels, m_texref));
        return py::make_tuple(fmt, num_channels);
      }
#endif

      unsigned int get_flags()
      {
        unsigned int result;
        MUSAPP_CALL_GUARDED(muTexRefGetFlags, (&result, m_texref));
        return result;
      }
  };

  // }}}

  // {{{ surface reference
#if MUSAPP_MUSA_VERSION >= 3010
  class module;

  class surface_reference : public  boost::noncopyable
  {
    private:
      MUsurfref m_surfref;

      // life support for array and module
      boost::shared_ptr<array> m_array;
      boost::shared_ptr<module> m_module;

    public:
      surface_reference(MUsurfref sr)
        : m_surfref(sr)
      { }

      void set_module(boost::shared_ptr<module> mod)
      { m_module = mod; }

      MUsurfref handle() const
      { return m_surfref; }

      void set_array(boost::shared_ptr<array> ary, unsigned int flags)
      {
        MUSAPP_CALL_GUARDED(muSurfRefSetArray, (m_surfref, ary->handle(), flags));
        m_array = ary;
      }

      array *get_array()
      {
        MUarray result;
        MUSAPP_CALL_GUARDED(muSurfRefGetArray, (&result, m_surfref));
        return new array(result, false);
      }
  };
#endif

  // }}}

  // {{{ module
  class function;

  class module : public boost::noncopyable, public context_dependent
  {
    private:
      MUmodule m_module;

    public:
      module(MUmodule mod)
        : m_module(mod)
      { }

      ~module()
      {
        try
        {
          scoped_context_activation ca(get_context());
          MUSAPP_CALL_GUARDED_CLEANUP(muModuleUnload, (m_module));
        }
        CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(module);
      }

      MUmodule handle() const
      { return m_module; }

      function get_function(const char *name);
      py::tuple get_global(const char *name)
      {
        MUdeviceptr devptr;
        pymusa_size_t bytes;
        MUSAPP_CALL_GUARDED(muModuleGetGlobal, (&devptr, &bytes, m_module, name));
        return py::make_tuple(devptr, bytes);
      }
  };

  inline
  module *module_from_file(const char *filename)
  {
    MUmodule mod;
    MUSAPP_CALL_GUARDED(muModuleLoad, (&mod, filename));
    return new module(mod);
  }

  inline
  texture_reference *module_get_texref(
      boost::shared_ptr<module> mod, const char *name)
  {
    MUtexref tr;
    MUSAPP_CALL_GUARDED(muModuleGetTexRef, (&tr, mod->handle(), name));
    std::unique_ptr<texture_reference> result(
        new texture_reference(tr, false));
    result->set_module(mod);
    return result.release();
  }

#if MUSAPP_MUSA_VERSION >= 3010
  inline
  surface_reference *module_get_surfref(
      boost::shared_ptr<module> mod, const char *name)
  {
    MUsurfref sr;
    MUSAPP_CALL_GUARDED(muModuleGetSurfRef, (&sr, mod->handle(), name));
    std::unique_ptr<surface_reference> result(
        new surface_reference(sr));
    result->set_module(mod);
    return result.release();
  }
#endif

  // }}}

  // {{{ function
  class function
  {
    private:
      MUfunction m_function;
      std::string m_symbol;

    public:
      function(MUfunction func, std::string const &sym)
        : m_function(func), m_symbol(sym)
      { }

      void set_block_shape(int x, int y, int z)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muFuncSetBlockShape, (m_function, x, y, z), m_symbol);
      }
      void set_shared_size(unsigned int bytes)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muFuncSetSharedSize, (m_function, bytes), m_symbol);
      }

      void param_set_size(unsigned int bytes)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muParamSetSize, (m_function, bytes), m_symbol);
      }
      void param_set(int offset, unsigned int value)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muParamSeti, (m_function, offset, value), m_symbol);
      }
      void param_set(int offset, float value)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
          muParamSetf, (m_function, offset, value), m_symbol);
      }
      void param_setv(int offset, void *buf, size_t len)
      {
        // maybe the unsigned int will change, it does not seem right
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
          muParamSetv, (m_function, offset, buf, (unsigned int) len), m_symbol);
      }
      void param_set_texref(const texture_reference &tr)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(muParamSetTexRef, (m_function,
            MU_PARAM_TR_DEFAULT, tr.handle()), m_symbol);
      }

      void launch()
      {
        MUSAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(
            muLaunch, (m_function), m_symbol);
      }
      void launch_grid(int grid_width, int grid_height)
      {
        MUSAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(
          muLaunchGrid, (m_function, grid_width, grid_height), m_symbol);
      }
      void launch_grid_async(int grid_width, int grid_height, const stream &s)
      {
        MUSAPP_CALL_GUARDED_THREADED_WITH_TRACE_INFO(
            muLaunchGridAsync, (m_function, grid_width, grid_height, s.handle()),
            m_symbol);
      }

#if MUSAPP_MUSA_VERSION >= 2020
      int get_attribute(MUfunction_attribute attr) const
      {
        int result;
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muFuncGetAttribute, (&result, attr, m_function), m_symbol);
        return result;
      }
#endif

      // 10.0+ is a guess
#if MUSAPP_MUSA_VERSION >= 10000
      void set_attribute(MUfunction_attribute attr, int value) const
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muFuncSetAttribute, (m_function, attr, value), m_symbol);
      }
#endif

#if MUSAPP_MUSA_VERSION >= 3000 && defined(MUSAPP_POST_30_BETA)
      void set_cache_config(MUfunc_cache fc)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muFuncSetCacheConfig, (m_function, fc), m_symbol);
      }
#endif

#if MUSAPP_MUSA_VERSION >= 4000
      void launch_kernel(py::tuple grid_dim_py, py::tuple block_dim_py,
          py::object parameter_buffer,
          unsigned shared_mem_bytes, py::object stream_py)
      {
        const unsigned axis_count = 3;
        unsigned grid_dim[axis_count];
        unsigned block_dim[axis_count];

        for (unsigned i = 0; i < axis_count; ++i)
        {
          grid_dim[i] = 1;
          block_dim[i] = 1;
        }

        pymusa_size_t gd_length = py::len(grid_dim_py);
        if (gd_length > axis_count)
          throw pymusa::error("function::launch_kernel", MUSA_ERROR_INVALID_HANDLE,
              "too many grid dimensions in kernel launch");

        for (unsigned i = 0; i < gd_length; ++i)
          grid_dim[i] = py::extract<unsigned>(grid_dim_py[i]);

        pymusa_size_t bd_length = py::len(block_dim_py);
        if (bd_length > axis_count)
          throw pymusa::error("function::launch_kernel", MUSA_ERROR_INVALID_HANDLE,
              "too many block dimensions in kernel launch");

        for (unsigned i = 0; i < bd_length; ++i)
          block_dim[i] = py::extract<unsigned>(block_dim_py[i]);

        PYMUSA_PARSE_STREAM_PY;

        py_buffer_wrapper par_buf_wrapper;
        par_buf_wrapper.get(parameter_buffer.ptr(), PyBUF_ANY_CONTIGUOUS);
        size_t par_len = par_buf_wrapper.m_buf.len;

        void *config[] = {
          MU_LAUNCH_PARAM_BUFFER_POINTER, const_cast<void *>(par_buf_wrapper.m_buf.buf),
          MU_LAUNCH_PARAM_BUFFER_SIZE, &par_len,
          MU_LAUNCH_PARAM_END
        };

        MUSAPP_CALL_GUARDED(
            muLaunchKernel, (m_function,
              grid_dim[0], grid_dim[1], grid_dim[2],
              block_dim[0], block_dim[1], block_dim[2],
              shared_mem_bytes, s_handle, 0, config
              ));
      }

#endif

#if MUSAPP_MUSA_VERSION >= 4020
      void set_shared_config(MUsharedconfig config)
      {
        MUSAPP_CALL_GUARDED_WITH_TRACE_INFO(
            muFuncSetSharedMemConfig, (m_function, config), m_symbol);
      }
#endif

  };

  inline
  function module::get_function(const char *name)
  {
    MUfunction func;
    MUSAPP_CALL_GUARDED(muModuleGetFunction, (&func, m_module, name));
    return function(func, name);
  }

  // }}}

  // {{{ device memory
  inline
  py::tuple mem_get_info()
  {
    pymusa_size_t free, total;
    MUSAPP_CALL_GUARDED(muMemGetInfo, (&free, &total));
    return py::make_tuple(free, total);
  }

  inline
  MUdeviceptr mem_alloc(size_t bytes)
  {
    MUdeviceptr devptr;
    MUSAPP_CALL_GUARDED(muMemAlloc, (&devptr, bytes));
    return devptr;
  }

  inline
  void mem_free(MUdeviceptr devptr)
  {
    MUSAPP_CALL_GUARDED_CLEANUP(muMemFree, (devptr));
  }

  // A class the user can override to make device_allocation-
  // workalikes.

  class pointer_holder_base
  {
    public:
      virtual ~pointer_holder_base() { }
      virtual MUdeviceptr get_pointer() const = 0;

      operator MUdeviceptr() const
      { return get_pointer(); }

      py::object as_buffer(size_t size, size_t offset)
      {
        return py::object(
            py::handle<>(
#if PY_VERSION_HEX >= 0x03030000
              PyMemoryView_FromMemory((char *) (get_pointer() + offset), size,
                PyBUF_WRITE)
#else /* Py2 */
              PyBuffer_FromReadWriteMemory((void *) (get_pointer() + offset), size)
#endif
              ));
      }
  };

  class device_allocation : public boost::noncopyable, public context_dependent
  {
    private:
      bool m_valid;

    protected:
      MUdeviceptr m_devptr;

    public:
      device_allocation(MUdeviceptr devptr)
        : m_valid(true), m_devptr(devptr)
      { }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_free(m_devptr);
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(device_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pymusa::error("device_allocation::free", MUSA_ERROR_INVALID_HANDLE);
      }

      ~device_allocation()
      {
        if (m_valid)
          free();
      }

      operator MUdeviceptr() const
      { return m_devptr; }

      py::object as_buffer(size_t size, size_t offset)
      {
        return py::object(
            py::handle<>(
#if PY_VERSION_HEX >= 0x03030000
              PyMemoryView_FromMemory((char *) (m_devptr + offset), size,
                PyBUF_WRITE)
#else /* Py2 */
              PyBuffer_FromReadWriteMemory((void *) (m_devptr + offset), size)
#endif
              ));
      }
  };

  inline Py_ssize_t mem_alloc_pitch(
      std::unique_ptr<device_allocation> &da,
        unsigned int width, unsigned int height, unsigned int access_size)
  {
    MUdeviceptr devptr;
    pymusa_size_t pitch;
    MUSAPP_CALL_GUARDED(muMemAllocPitch, (&devptr, &pitch, width, height, access_size));
    da = std::unique_ptr<device_allocation>(new device_allocation(devptr));
    return pitch;
  }

  inline
  py::tuple mem_get_address_range(MUdeviceptr ptr)
  {
    MUdeviceptr base;
    pymusa_size_t size;
    MUSAPP_CALL_GUARDED(muMemGetAddressRange, (&base, &size, ptr));
    return py::make_tuple(base, size);
  }

  inline
  void memcpy_dtoa(array const &ary, unsigned int index, MUdeviceptr src, size_t len)
  { MUSAPP_CALL_GUARDED_THREADED(muMemcpyDtoA, (ary.handle(), index, src, len)); }

  inline
  void memcpy_atod(MUdeviceptr dst, array const &ary, unsigned int index, size_t len)
  { MUSAPP_CALL_GUARDED_THREADED(muMemcpyAtoD, (dst, ary.handle(), index, len)); }

  inline
  void memcpy_atoa(
      array const &dst, unsigned int dst_index,
      array const &src, unsigned int src_index,
      size_t len)
  { MUSAPP_CALL_GUARDED_THREADED(muMemcpyAtoA, (dst.handle(), dst_index, src.handle(), src_index, len)); }

  // }}}

  // {{{ ipc_mem_handle

#if MUSAPP_MUSA_VERSION >= 4010 && PY_VERSION_HEX >= 0x02060000
  class ipc_mem_handle : public boost::noncopyable, public context_dependent
  {
    private:
      bool m_valid;

    protected:
      MUdeviceptr m_devptr;

    public:
      ipc_mem_handle(py::object obj, MUipcMem_flags flags=MU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        : m_valid(true)
      {
        if (!PyByteArray_Check(obj.ptr()))
          throw pymusa::error("event_from_ipc_handle", MUSA_ERROR_INVALID_VALUE,
              "argument is not a bytes array");
        MUipcMemHandle handle;
        if (PyByteArray_GET_SIZE(obj.ptr()) != sizeof(handle))
          throw pymusa::error("event_from_ipc_handle", MUSA_ERROR_INVALID_VALUE,
              "handle has the wrong size");
        memcpy(&handle, PyByteArray_AS_STRING(obj.ptr()), sizeof(handle));

        MUSAPP_CALL_GUARDED(muIpcOpenMemHandle, (&m_devptr, handle, flags));
      }

      void close()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            MUSAPP_CALL_GUARDED_CLEANUP(muIpcCloseMemHandle, (m_devptr));
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(ipc_mem_handle);

          release_context();
          m_valid = false;
        }
        else
          throw pymusa::error("ipc_mem_handle::close", MUSA_ERROR_INVALID_HANDLE);
      }

      ~ipc_mem_handle()
      {
        if (m_valid)
          close();
      }

      operator MUdeviceptr() const
      { return m_devptr; }
  };

  inline
  py::object mem_get_ipc_handle(MUdeviceptr devptr)
  {
    MUipcMemHandle handle;
    MUSAPP_CALL_GUARDED(muIpcGetMemHandle, (&handle, devptr));
    return py::object(py::handle<>(PyByteArray_FromStringAndSize(
            reinterpret_cast<const char *>(&handle),
            sizeof(handle))));
  }

#endif

  // }}}

  // {{{ structured memcpy

#define MEMCPY_SETTERS \
    void set_src_host(py::object buf_py) \
    { \
      srcMemoryType = MU_MEMORYTYPE_HOST; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_STRIDED_RO); \
      srcHost = buf_wrapper.m_buf.buf; \
    } \
    \
    void set_src_array(array const &ary)  \
    {  \
      srcMemoryType = MU_MEMORYTYPE_ARRAY; \
      srcArray = ary.handle();  \
    } \
    \
    void set_src_device(MUdeviceptr devptr)  \
    { \
      srcMemoryType = MU_MEMORYTYPE_DEVICE; \
      srcDevice = devptr; \
    } \
    \
    void set_dst_host(py::object buf_py) \
    { \
      dstMemoryType = MU_MEMORYTYPE_HOST; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_STRIDED); \
      dstHost = buf_wrapper.m_buf.buf; \
    } \
    \
    void set_dst_array(array const &ary) \
    { \
      dstMemoryType = MU_MEMORYTYPE_ARRAY; \
      dstArray = ary.handle(); \
    } \
    \
    void set_dst_device(MUdeviceptr devptr)  \
    { \
      dstMemoryType = MU_MEMORYTYPE_DEVICE; \
      dstDevice = devptr; \
    }

#if MUSAPP_MUSA_VERSION >= 4000
#define MEMCPY_SETTERS_UNIFIED \
    void set_src_unified(py::object buf_py) \
    { \
      srcMemoryType = MU_MEMORYTYPE_UNIFIED; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_ANY_CONTIGUOUS); \
      srcHost = buf_wrapper.m_buf.buf; \
    } \
    \
    void set_dst_unified(py::object buf_py) \
    { \
      dstMemoryType = MU_MEMORYTYPE_UNIFIED; \
      py_buffer_wrapper buf_wrapper; \
      buf_wrapper.get(buf_py.ptr(), PyBUF_ANY_CONTIGUOUS | PyBUF_WRITABLE); \
      dstHost = buf_wrapper.m_buf.buf; \
    }
#else
#define MEMCPY_SETTERS_UNIFIED /* empty */
#endif





  struct memcpy_2d : public MUSA_MEMCPY2D
  {
    memcpy_2d() : MUSA_MEMCPY2D()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void execute(bool aligned=false) const
    {
      if (aligned)
      { MUSAPP_CALL_GUARDED_THREADED(muMemcpy2D, (this)); }
      else
      { MUSAPP_CALL_GUARDED_THREADED(muMemcpy2DUnaligned, (this)); }
    }

    void execute_async(const stream &s) const
    { MUSAPP_CALL_GUARDED_THREADED(muMemcpy2DAsync, (this, s.handle())); }
  };

#if MUSAPP_MUSA_VERSION >= 2000
  struct memcpy_3d : public MUSA_MEMCPY3D
  {
    memcpy_3d() : MUSA_MEMCPY3D()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void execute() const
    {
      MUSAPP_CALL_GUARDED_THREADED(muMemcpy3D, (this));
    }

    void execute_async(const stream &s) const
    { MUSAPP_CALL_GUARDED_THREADED(muMemcpy3DAsync, (this, s.handle())); }
  };
#endif

#if MUSAPP_MUSA_VERSION >= 4000
  struct memcpy_3d_peer : public MUSA_MEMCPY3D_PEER
  {
    memcpy_3d_peer() : MUSA_MEMCPY3D_PEER()
    {
    }

    MEMCPY_SETTERS;
    MEMCPY_SETTERS_UNIFIED;

    void set_src_context(context const &ctx)
    {
      srcContext = ctx.handle();
    }

    void set_dst_context(context const &ctx)
    {
      dstContext = ctx.handle();
    }

    void execute() const
    {
      MUSAPP_CALL_GUARDED_THREADED(muMemcpy3DPeer, (this));
    }

    void execute_async(const stream &s) const
    { MUSAPP_CALL_GUARDED_THREADED(muMemcpy3DPeerAsync, (this, s.handle())); }
  };
#endif

  // }}}

  // {{{ host memory
  inline void *mem_host_alloc(size_t size, unsigned flags=0)
  {
    void *m_data;
#if MUSAPP_MUSA_VERSION >= 2020
    MUSAPP_CALL_GUARDED(muMemHostAlloc, (&m_data, size, flags));
#else
    if (flags != 0)
      throw pymusa::error("mem_host_alloc", MUSA_ERROR_INVALID_VALUE,
          "nonzero flags in mem_host_alloc not allowed in CUDA 2.1 and older");
    MUSAPP_CALL_GUARDED(muMemAllocHost, (&m_data, size));
#endif
    return m_data;
  }

  inline void mem_host_free(void *ptr)
  {
    MUSAPP_CALL_GUARDED_CLEANUP(muMemFreeHost, (ptr));
  }

#if MUSAPP_MUSA_VERSION >= 6000
  inline MUdeviceptr mem_managed_alloc(size_t size, unsigned flags=0)
  {
    MUdeviceptr m_data;
    MUSAPP_CALL_GUARDED(muMemAllocManaged, (&m_data, size, flags));
    return m_data;
  }
#endif

#if MUSAPP_MUSA_VERSION >= 4000
  inline void *mem_host_register(void *ptr, size_t bytes, unsigned int flags=0)
  {
    MUSAPP_CALL_GUARDED(muMemHostRegister, (ptr, bytes, flags));
    return ptr;
  }

  inline void mem_host_unregister(void *ptr)
  {
    MUSAPP_CALL_GUARDED_CLEANUP(muMemHostUnregister, (ptr));
  }
#endif

  inline void *aligned_malloc(size_t size, size_t alignment, void **original_pointer)
  {
    // alignment must be a power of two.
    if ((alignment & (alignment - 1)) != 0)
      throw pymusa::error("aligned_malloc", MUSA_ERROR_INVALID_VALUE,
          "alignment must be a power of two");

    if (alignment == 0)
      throw pymusa::error("aligned_malloc", MUSA_ERROR_INVALID_VALUE,
          "alignment must non-zero");

    void *p = malloc(size + (alignment - 1));
    if (!p)
      throw pymusa::error("aligned_malloc", MUSA_ERROR_OUT_OF_MEMORY,
          "aligned malloc failed");

    *original_pointer = p;

    p = (void *)((((ptrdiff_t)(p)) + (alignment-1)) & -alignment);
    return p;
  }



  struct host_pointer : public boost::noncopyable, public context_dependent
  {
    protected:
      bool m_valid;
      void *m_data;

    public:
      host_pointer()
        : m_valid(false)
      { }

      host_pointer(void *ptr)
        : m_valid(true), m_data(ptr)
      { }

      virtual ~host_pointer()
      { }

      void *data()
      { return m_data; }

#if MUSAPP_MUSA_VERSION >= 2020
      MUdeviceptr get_device_pointer()
      {
        MUdeviceptr result;
        MUSAPP_CALL_GUARDED(muMemHostGetDevicePointer, (&result, m_data, 0));
        return result;
      }
#endif

  };

  struct pagelocked_host_allocation : public host_pointer
  {
    public:
      pagelocked_host_allocation(size_t bytesize, unsigned flags=0)
        : host_pointer(mem_host_alloc(bytesize, flags))
      { }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~pagelocked_host_allocation()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_host_free(m_data);
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(pagelocked_host_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pymusa::error("pagelocked_host_allocation::free", MUSA_ERROR_INVALID_HANDLE);
      }

#if MUSAPP_MUSA_VERSION >= 3020
      unsigned int get_flags()
      {
        unsigned int flags;
        MUSAPP_CALL_GUARDED(muMemHostGetFlags, (&flags, m_data));
        return flags;
      }
#endif
  };

  struct aligned_host_allocation : public host_pointer
  {
      void *m_original_pointer;

    public:
      aligned_host_allocation(size_t size, size_t alignment)
        : host_pointer(aligned_malloc(size, alignment, &m_original_pointer))
      { }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~aligned_host_allocation()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          ::free(m_original_pointer);
          m_valid = false;
        }
        else
          throw pymusa::error("aligned_host_allocation::free", MUSA_ERROR_INVALID_HANDLE);
      }
  };

#if MUSAPP_MUSA_VERSION >= 6000
  struct managed_allocation : public device_allocation
  {
    public:
      managed_allocation(size_t bytesize, unsigned flags=0)
        : device_allocation(mem_managed_alloc(bytesize, flags))
      { }

      // The device pointer is also valid on the host
      void *data()
      { return (void *) m_devptr; }

      MUdeviceptr get_device_pointer()
      {
        return m_devptr;
      }

      void attach(unsigned flags, py::object stream_py)
      {
        PYMUSA_PARSE_STREAM_PY;

        MUSAPP_CALL_GUARDED(muStreamAttachMemAsync, (s_handle, m_devptr, 0, flags));
      }

  };
#endif


#if MUSAPP_MUSA_VERSION >= 4000
  struct registered_host_memory : public host_pointer
  {
    private:
      py::object m_base;

    public:
      registered_host_memory(void *p, size_t bytes, unsigned int flags=0,
          py::object base=py::object())
        : host_pointer(mem_host_register(p, bytes, flags)), m_base(base)
      {
      }

      /* Don't try to be clever and coalesce these in the base class.
       * Won't work: Destructors may not call virtual functions.
       */
      ~registered_host_memory()
      {
        if (m_valid)
          free();
      }

      void free()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            mem_host_unregister(m_data);
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(host_allocation);

          release_context();
          m_valid = false;
        }
        else
          throw pymusa::error("registered_host_memory::free", MUSA_ERROR_INVALID_HANDLE);
      }

      py::object base() const
      {
        return m_base;
      }
  };
#endif

  // }}}

  // {{{ event
  class event : public boost::noncopyable, public context_dependent
  {
    private:
      MUevent m_event;

    public:
      event(unsigned int flags=0)
      { MUSAPP_CALL_GUARDED(muEventCreate, (&m_event, flags)); }

      event(MUevent evt)
      : m_event(evt)
      { }

      ~event()
      {
        try
        {
          scoped_context_activation ca(get_context());
          MUSAPP_CALL_GUARDED_CLEANUP(muEventDestroy, (m_event));
        }
        CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(event);
      }

      event *record(py::object stream_py)
      {
        PYMUSA_PARSE_STREAM_PY;

        MUSAPP_CALL_GUARDED(muEventRecord, (m_event, s_handle));
        return this;
      }

      MUevent handle() const
      { return m_event; }

      event *synchronize()
      {
        MUSAPP_CALL_GUARDED_THREADED(muEventSynchronize, (m_event));
        return this;
      }

      bool query() const
      {
        MUSAPP_PRINT_CALL_TRACE("muEventQuery");

        MUresult result = muEventQuery(m_event);
        switch (result)
        {
          case MUSA_SUCCESS:
            return true;
          case MUSA_ERROR_NOT_READY:
            return false;
          default:
            CUDAPP_PRINT_ERROR_TRACE("muEventQuery", result);
            throw error("muEventQuery", result);
        }
      }

      float time_since(event const &start)
      {
        float result;
        MUSAPP_CALL_GUARDED(muEventElapsedTime, (&result, start.m_event, m_event));
        return result;
      }

      float time_till(event const &end)
      {
        float result;
        MUSAPP_CALL_GUARDED(muEventElapsedTime, (&result, m_event, end.m_event));
        return result;
      }

#if MUSAPP_MUSA_VERSION >= 4010 && PY_VERSION_HEX >= 0x02060000
      py::object ipc_handle()
      {
        MUipcEventHandle handle;
        MUSAPP_CALL_GUARDED(muIpcGetEventHandle, (&handle, m_event));
        return py::object(py::handle<>(PyByteArray_FromStringAndSize(
              reinterpret_cast<const char *>(&handle),
              sizeof(handle))));
      }
#endif
  };

#if MUSAPP_MUSA_VERSION >= 3020
  inline void stream::wait_for_event(const event &evt)
  {
    MUSAPP_CALL_GUARDED(muStreamWaitEvent, (m_stream, evt.handle(), 0));
  }
#endif

#if MUSAPP_MUSA_VERSION >= 4010 && PY_VERSION_HEX >= 0x02060000
  inline
  event *event_from_ipc_handle(py::object obj)
  {
    if (!PyByteArray_Check(obj.ptr()))
      throw pymusa::error("event_from_ipc_handle", MUSA_ERROR_INVALID_VALUE,
          "argument is not a bytes array");
    MUipcEventHandle handle;
    if (PyByteArray_GET_SIZE(obj.ptr()) != sizeof(handle))
      throw pymusa::error("event_from_ipc_handle", MUSA_ERROR_INVALID_VALUE,
          "handle has the wrong size");
    memcpy(&handle, PyByteArray_AS_STRING(obj.ptr()), sizeof(handle));

    MUevent evt;
    MUSAPP_CALL_GUARDED(muIpcOpenEventHandle, (&evt, handle));

    return new event(evt);
  }
#endif

  // }}}

  // {{{ profiler
#if MUSAPP_MUSA_VERSION >= 4000
  inline void initialize_profiler(
      const char *config_file,
      const char *output_file,
      MUoutput_mode output_mode)
  {
    MUSAPP_CALL_GUARDED(muProfilerInitialize, (config_file, output_file, output_mode));
  }

  inline void start_profiler()
  {
    MUSAPP_CALL_GUARDED(muProfilerStart, ());
  }

  inline void stop_profiler()
  {
    MUSAPP_CALL_GUARDED(muProfilerStop, ());
  }
#endif
  // }}}
}




#endif
// vim: foldmethod=marker
