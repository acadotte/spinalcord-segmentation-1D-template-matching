# CMake file for generating Python modules.

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

FIND_PACKAGE(ITK REQUIRED)
INCLUDE(${ITK_USE_FILE})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "-builtin")

# example C++ library 
ADD_LIBRARY(calculus calculus.cpp calculus.h)
TARGET_LINK_LIBRARIES(calculus)

# ITK library 
LINK_LIBRARIES(calculus ITKCommon ITKIO ITKBasicFilters)

# calculus Python module
SET_SOURCE_FILES_PROPERTIES(calculus.i PROPERTIES CPLUSPLUS ON)
SWIG_ADD_MODULE(calculus python calculus.i calculus.cpp calculus.h)
SWIG_LINK_LIBRARIES(calculus ${PYTHON_LIBRARIES})

