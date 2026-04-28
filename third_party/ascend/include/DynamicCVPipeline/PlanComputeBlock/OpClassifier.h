/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_OP_CLASSIFIER_H
#define TRITON_ADAPTER_OP_CLASSIFIER_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

// Core type enumeration for operations
enum OpCoreType { OP_UNDETERMINED = 0, OP_CUBE_ONLY = 1, OP_VECTOR_ONLY = 2, OP_CUBE_AND_VECTOR = 3 };

// OpClassifierPass for categorizing operations as CUBE or VECTOR
class OpClassifierPass : public PassWrapper<OpClassifierPass, OperationPass<ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpClassifierPass)

    // Constructor
    OpClassifierPass() = default;

    // Run the pass
    void runOnOperation();

private:
    // Map from operation to its core type
    llvm::DenseMap<Operation *, OpCoreType> opCoreTypes;

    // All operations in the module
    llvm::SmallVector<Operation *> allOps;

    // Seed operations for CUBE upstream propagation
    llvm::SmallVector<Operation *> cubeSeeds;

    std::shared_ptr<AliasAnalysis> aliasAnalysis;
    std::shared_ptr<MemoryDependenceGraph> memDepGraph;

    // Mark an operation as CUBE
    void markCube(Operation *op);

    // Pattern matching for CUBE operations
    int patternMatchCUBE();

    // Upstream pattern matching helpers
    void matchToTensorPattern(Operation *def);
    void matchTransposePattern(Operation *def);
    void matchFillPattern(Operation *def);

    // Downstream pattern matching helpers
    void matchStorePattern(Operation *user);
    void matchExtractSlicePattern(Operation *user);
    void matchMaterializePattern(Operation *user);

    // Propagate CUBE core type upstream
    int propagateCubeUpstream();

    // Get upstream operations based on both SSA and memory dependencies
    void getUpstreamOpsWithMemoryDeps(Operation *cur, llvm::SmallVectorImpl<Operation *> &upstreamOps);
};

// Create the pass
std::unique_ptr<OperationPass<ModuleOp>> createOpClassifierPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_OP_CLASSIFIER_H
