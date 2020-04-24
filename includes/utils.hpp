#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <iostream>


#define CHECK_EQ(val1, val2, message)                              \
  do {                                                             \
    if (val1 != val2) {                                            \
      std::cerr << __FILE__ << "(" << __LINE__ << "): " << message \
                << std::endl;                                      \
      exit(1);                                                     \
    }                                                              \
  } while (0)


#define INIT_STORAGE(storage_ptr, shape)            \
  do {                                              \
    if (storage_ptr.get() == nullptr) {             \
      storage_ptr.reset(new Container(shape));        \
    } else if (storage_ptr->get_shape() != shape) { \
      storage_ptr->resize(shape);                   \
    }                                               \
  } while (0)