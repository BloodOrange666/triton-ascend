file(GLOB PLAN_COMPUTE_BLOCK_SRC_FILES CONFIGURE_DEPENDS
    "PlanComputeBlock/*.cpp"
)

add_triton_library(DynamicCVPipeline
  AddDynamicCVPipeline.cpp
  PlanComputeBlock.cpp
  SplitDataflow.cpp
  SeparateMemoryFromComputePass.cpp
  AllocMultiCache.cpp
  AddControlFlowCondition.cpp
  ${PLAN_COMPUTE_BLOCK_SRC_FILES}

  DEPENDS
  AddDynamicCVPipelinePassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
)