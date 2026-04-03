#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pymusa_ARRAY_API

#include "musa.hpp"

boost::thread_specific_ptr<pymusa::context_stack> pymusa::context_stack_ptr;
