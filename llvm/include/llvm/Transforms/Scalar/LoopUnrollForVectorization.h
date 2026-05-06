//===- LoopUnrollForVectorization.h - Unroll inner loops for vectorization
//-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PROTOTYPE PASS: This is an experimental pass under active development.
//
// This pass fully unrolls small inner loops within outer loop nests that have
// explicit vectorization hints (llvm.loop.vectorize.enable).
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_LOOPUNROLLFORVECTORIZATION_H
#define LLVM_TRANSFORMS_SCALAR_LOOPUNROLLFORVECTORIZATION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Function;

class LoopUnrollForVectorizationPass
    : public PassInfoMixin<LoopUnrollForVectorizationPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_SCALAR_LOOPUNROLLFORVECTORIZATION_H
