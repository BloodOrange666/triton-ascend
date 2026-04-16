add_triton_library(DynamicCVPipeline
  AddDynamicCVPipeline.cpp

  DEPENDS
  AddDynamicCVPipelinePassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
)