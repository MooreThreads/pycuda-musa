#ifndef _AFJDFJSDFSD_PYMUSA_HEADER_SEEN_CUDA_GL_HPP
#define _AFJDFJSDFSD_PYMUSA_HEADER_SEEN_CUDA_GL_HPP



#include <musa.hpp>
#if defined(__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#else  /* __APPLE__ */
#include <GL/gl.h>
#endif 

#include <musaGL.h>


namespace pymusa { namespace gl {

  // {{{ pre-3.0-style API

  inline
  void gl_init()
  {
    MUSAPP_CALL_GUARDED(muGLInit, ());
    PyErr_Warn(
        PyExc_DeprecationWarning,
        "gl_init() has been deprecated since CUDA 3.0 "
        "and PyCUDA 2011.1.");
  }




  inline
  boost::shared_ptr<context> make_gl_context(device const &dev, unsigned int flags)
  {
    MUcontext ctx;
    MUSAPP_CALL_GUARDED(muGLCtxCreate, (&ctx, flags, dev.handle()));
    boost::shared_ptr<context> result(new context(ctx));
    context_stack::get().push(result);
    return result;
  }




  class buffer_object : public context_dependent
  {
    private:
      GLuint m_handle;
      bool m_valid;

    public:
      buffer_object(GLuint handle)
        : m_handle(handle), m_valid(true)
      {
        MUSAPP_CALL_GUARDED(muGLRegisterBufferObject, (handle));
        PyErr_Warn(
            PyExc_DeprecationWarning,
            "buffer_object has been deprecated since CUDA 3.0 "
            "and PyCUDA 2011.1.");
      }

      ~buffer_object()
      {
        if (m_valid)
          unregister();
      }

      GLuint handle()
      { return m_handle; }

      void unregister()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            MUSAPP_CALL_GUARDED_CLEANUP(muGLUnregisterBufferObject, (m_handle));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(buffer_object);
        }
        else
          throw pymusa::error("buffer_object::unregister", MUSA_ERROR_INVALID_HANDLE);
      }
  };



  class buffer_object_mapping : public context_dependent
  {
    private:
      boost::shared_ptr<buffer_object> m_buffer_object;
      MUdeviceptr m_devptr;
      size_t m_size;
      bool m_valid;

    public:
      buffer_object_mapping(
          boost::shared_ptr<buffer_object> bobj,
          MUdeviceptr devptr,
          size_t size)
        : m_buffer_object(bobj), m_devptr(devptr), m_size(size), m_valid(true)
      { 
        PyErr_Warn(
            PyExc_DeprecationWarning,
            "buffer_object_mapping has been deprecated since CUDA 3.0 "
            "and PyCUDA 2011.1.");
      }

      ~buffer_object_mapping()
      {
        if (m_valid)
          unmap();
      }

      void unmap()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            MUSAPP_CALL_GUARDED_CLEANUP(muGLUnmapBufferObject, (m_buffer_object->handle()));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(buffer_object_mapping)
        }
        else
          throw pymusa::error("buffer_object_mapping::unmap", MUSA_ERROR_INVALID_HANDLE);
      }

      MUdeviceptr device_ptr() const
      { return m_devptr; }

      size_t size() const
      { return m_size; }
  };




  inline buffer_object_mapping *map_buffer_object(
      boost::shared_ptr<buffer_object> bobj)
  {
    MUdeviceptr devptr;
    pymusa_size_t size;
    MUSAPP_CALL_GUARDED(muGLMapBufferObject, (&devptr, &size, bobj->handle()));
    PyErr_Warn(
        PyExc_DeprecationWarning,
        "map_buffer_object has been deprecated since CUDA 3.0 "
        "and PyCUDA 2011.1.");

    return new buffer_object_mapping(bobj, devptr, size);
  }

  // }}}

  // {{{ new-style (3.0+) API

#if MUSAPP_MUSA_VERSION >= 3000
  class registered_object : public context_dependent
  {
    protected:
      GLuint m_gl_handle;
      bool m_valid;
      MUgraphicsResource m_resource;

    public:
      registered_object(GLuint gl_handle)
        : m_gl_handle(gl_handle), m_valid(true)
      {
      }

      ~registered_object()
      {
        if (m_valid)
          unregister();
      }

      GLuint gl_handle()
      { return m_gl_handle; }

      MUgraphicsResource resource()
      { return m_resource; }

      void unregister()
      {
        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            MUSAPP_CALL_GUARDED_CLEANUP(
                muGraphicsUnregisterResource, (m_resource));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(registered_object);
        }
        else
          throw pymusa::error("registered_object::unregister", 
              MUSA_ERROR_INVALID_HANDLE);
      }
  };

  class registered_buffer : public registered_object
  {
    public:
      registered_buffer(GLuint gl_handle, 
          MUgraphicsMapResourceFlags flags=MU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
        : registered_object(gl_handle)
      {
        MUSAPP_CALL_GUARDED(muGraphicsGLRegisterBuffer, 
            (&m_resource, gl_handle, flags));
      }
  };

  class registered_image : public registered_object
  {
    public:
      registered_image(GLuint gl_handle, GLenum target, 
          MUgraphicsMapResourceFlags flags=MU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE)
        : registered_object(gl_handle)
      {
        MUSAPP_CALL_GUARDED(muGraphicsGLRegisterImage, 
            (&m_resource, gl_handle, target, flags));
      }
  };



  class registered_mapping : public context_dependent
  {
    private:
      boost::shared_ptr<registered_object> m_object;
      boost::shared_ptr<stream> m_stream;
      bool m_valid;

    public:
      registered_mapping(
          boost::shared_ptr<registered_object> robj,
          boost::shared_ptr<stream> strm)
        : m_object(robj), m_stream(strm), m_valid(true)
      { }

      ~registered_mapping()
      {
        if (m_valid)
          unmap_no_strm();
      }

      void unmap_no_strm()
      {
        unmap(m_stream);
      }

      void unmap(boost::shared_ptr<stream> const &strm)
      {
        MUstream s_handle;
        if (!strm.get())
          s_handle = 0;
        else
          s_handle = strm->handle();

        if (m_valid)
        {
          try
          {
            scoped_context_activation ca(get_context());
            MUgraphicsResource res = m_object->resource();
            MUSAPP_CALL_GUARDED_CLEANUP(muGraphicsUnmapResources,
                (1, &res, s_handle));
            m_valid = false;
          }
          CUDAPP_CATCH_CLEANUP_ON_DEAD_CONTEXT(registered_mapping)
        }
        else
          throw pymusa::error("registered_mapping::unmap", MUSA_ERROR_INVALID_HANDLE);
      }

      py::tuple device_ptr_and_size() const
      {
        MUdeviceptr devptr;
        pymusa_size_t size;
        MUSAPP_CALL_GUARDED(muGraphicsResourceGetMappedPointer, 
            (&devptr, &size, m_object->resource()));
        return py::make_tuple(devptr, size);
      }

      inline
      pymusa::array *array(unsigned int index, unsigned int level) const
      {
        MUarray devptr;
        MUSAPP_CALL_GUARDED(muGraphicsSubResourceGetMappedArray, 
            (&devptr, m_object->resource(), index, level));
        std::unique_ptr<pymusa::array> result(
            new pymusa::array(devptr, false));
        return result.release();
      }
  };




  inline registered_mapping *map_registered_object(
      boost::shared_ptr<registered_object> const &robj,
      py::object strm_py)
  {
    MUstream s_handle;
    boost::shared_ptr<stream> strm_sptr;

    if (strm_py.ptr() == Py_None)
    {
      s_handle = 0;
    }
    else
    {
      strm_sptr = py::extract<boost::shared_ptr<stream> >(strm_py);
      s_handle = strm_sptr->handle();
    }

    MUgraphicsResource res = robj->resource();
    MUSAPP_CALL_GUARDED(muGraphicsMapResources,
        (1, &res, s_handle));

    return new registered_mapping(robj, strm_sptr);
  }
#endif

  // }}}

} }




#endif

// vim: foldmethod=marker
