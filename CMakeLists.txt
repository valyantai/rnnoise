
cmake_minimum_required (VERSION 3.7.2)
project(rnnoise)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# add_definitions(-DFLOAT_APPROX)

set(GCC_COMPILE_FLAGS "-O3 -Wall -Wfloat-conversion -Wextra")
IF(WIN32)
  IF(MINGW)
    set(CMAKE_C_FLAGS "${GCC_COMPILE_FLAGS} -flto -funroll-loops -mfpmath=sse -fschedule-insns -fsched-pressure -m32 -march=atom")
  ENDIF()
ENDIF()
include_directories ("${PROJECT_SOURCE_DIR}/include")

OPTION(RNNOISE_LINK_WITH_OPUS "Set when statically linking with libopus" ON)
set(C_SOURCES
  src/denoise.c
  src/kiss_fft.c
  src/rnn_data.c
  src/rnn_reader.c
  src/rnn.c
)
IF(RNNOISE_LINK_WITH_OPUS)
  MESSAGE(STATUS "RNNoise will not build opus utilities.")
ELSE()
  MESSAGE(STATUS "RNNoise Building opus utilities.")

  set(C_SOURCES
    ${C_SOURCES}
    src/celt_lpc.c
    src/pitch.c
  )
ENDIF()

add_library(rnnoise STATIC ${C_SOURCES})
