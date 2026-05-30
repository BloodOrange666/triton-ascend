include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

add_triton_ut(
  NAME CostModelHardwareConfig
  SRCS HardwareConfigTest.cpp
  LIBS
    AscendModelAnalysis
)

add_triton_ut(
  NAME CostModelPipelineScheduler
  SRCS PipelineSchedulerTest.cpp
  LIBS
    AscendModelAnalysis
    AscendModelIR
)
