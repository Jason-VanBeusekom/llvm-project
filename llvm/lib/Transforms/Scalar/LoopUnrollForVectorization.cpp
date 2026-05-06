//===- LoopUnrollForVectorization.cpp - Unroll inner loops for outer vec
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass fully unrolls small-trip-count inner loops inside outer loop nests
// annotated with explicit vectorization hints (e.g., !dir$ vector always in
// Fortran, #pragma clang loop vectorize(enable) in C/C++).
//
// Motivation: Outer loop vectorization via the VPlan native path requires all
// header PHI nodes in the outer loop to be recognized as either induction
// variables or reductions at the outer loop level. When reduction variables
// flow through inner loop PHI nodes (e.g., a summation accumulated across a
// deeply nested loop), the analysis (RecurrenceDescriptor::AddReductionVar)
// cannot trace the reduction chain through inner loop PHIs.
//
// By fully unrolling small inner loops (as CCE's "unwind" pass does), the
// inner loop PHIs are eliminated, exposing the reduction at the outer loop
// level and enabling the outer loop vectorizer to proceed.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopUnrollForVectorization.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"

using namespace llvm;

#define DEBUG_TYPE "loop-unroll-for-vectorization"

static cl::opt<unsigned> UnrollForVecMaxTripCount(
    "unroll-for-vec-max-trip-count", cl::init(16), cl::Hidden,
    cl::desc("Maximum trip count of inner loops to unroll for outer loop "
             "vectorization"));

static cl::opt<unsigned> UnrollForVecMaxUnrolledSize(
    "unroll-for-vec-max-unrolled-size", cl::init(4096), cl::Hidden,
    cl::desc("Maximum total unrolled instruction count for inner loop "
             "unrolling to enable outer loop vectorization"));

/// Check whether a loop has explicit vectorization enable metadata.
static bool hasVectorizeEnableHint(const Loop *L) {
  MDNode *LoopID = L->getLoopID();
  if (!LoopID)
    return false;

  for (const MDOperand &MDO : drop_begin(LoopID->operands())) {
    const MDNode *MD = dyn_cast<MDNode>(MDO);
    if (!MD || MD->getNumOperands() < 2)
      continue;
    const MDString *S = dyn_cast<MDString>(MD->getOperand(0));
    if (!S)
      continue;
    if (S->getString() == "llvm.loop.vectorize.enable") {
      const ConstantAsMetadata *CM =
          dyn_cast<ConstantAsMetadata>(MD->getOperand(1));
      if (CM) {
        if (const ConstantInt *CI = dyn_cast<ConstantInt>(CM->getValue()))
          return CI->isOne();
      }
    }
  }
  return false;
}

/// Collect innermost sub-loops that are candidates for full unrolling.
/// Only collects currently-innermost loops with small constant trip counts.
/// After these are unrolled, their parents may become innermost and eligible
/// for subsequent collection.
static void
collectInnermostUnrollCandidates(Loop *Parent, ScalarEvolution &SE,
                                 SmallVectorImpl<Loop *> &Candidates) {
  for (Loop *Sub : *Parent) {
    if (Sub->isInnermost()) {
      unsigned TripCount = SE.getSmallConstantTripCount(Sub);
      if (TripCount > 0 && TripCount <= UnrollForVecMaxTripCount)
        Candidates.push_back(Sub);
    } else {
      collectInnermostUnrollCandidates(Sub, SE, Candidates);
    }
  }
}

/// Estimate the instruction count of a loop body (excluding the loop
/// overhead itself).
static unsigned estimateLoopBodySize(const Loop *L) {
  unsigned Size = 0;
  for (BasicBlock *BB : L->blocks())
    Size += BB->size();
  return Size;
}

static bool hasWorkExcludingSubloops(const Loop *L) {
  // Collect blocks that belong to sub-loops
  SmallPtrSet<BasicBlock *, 16> SubloopBlocks;
  for (Loop *Sub : *L)
    for (BasicBlock *BB : Sub->blocks())
      SubloopBlocks.insert(BB);

  // Check blocks that belong to L but not to any sub-loop
  for (BasicBlock *BB : L->blocks()) {
    if (SubloopBlocks.count(BB))
      continue;

    for (const Instruction &I : *BB) {
      // Skip loop overhead: PHIs, branches, and simple comparisons
      if (isa<PHINode>(I) || isa<BranchInst>(I) || isa<ICmpInst>(I))
        continue;

      return true;
    }
  }
  return false;
}

/// Recursively collect all loops in a nest and record which have their own
/// work (excluding sub-loops).
static void collectLoopsWithWork(Loop *L,
                                 SmallPtrSetImpl<BasicBlock *> &LoopsWithWork) {
  if (hasWorkExcludingSubloops(L))
    LoopsWithWork.insert(L->getHeader());
  for (Loop *Sub : *L)
    collectLoopsWithWork(Sub, LoopsWithWork);
}

PreservedAnalyses
LoopUnrollForVectorizationPass::run(Function &F, FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  bool Changed = false;

  // Only process loop nests where the user has explicitly requested
  // vectorization (e.g., !dir$ vector always → llvm.loop.vectorize.enable).
  // Walk the loop tree to find loops with the hint, then collect their
  // ancestors as the set of nests to process.
  SmallPtrSet<Loop *, 4> VecHintNests;
  for (Loop *TopLevel : LI) {
    SmallVector<Loop *, 8> Worklist;
    Worklist.push_back(TopLevel);
    while (!Worklist.empty()) {
      Loop *L = Worklist.pop_back_val();
      if (hasVectorizeEnableHint(L)) {
        // Mark this loop and all its ancestors as part of a vec-hint nest.
        for (Loop *Ancestor = L; Ancestor; Ancestor = Ancestor->getParentLoop())
          VecHintNests.insert(Ancestor);
      }
      for (Loop *Sub : *L)
        Worklist.push_back(Sub);
    }
  }

  if (VecHintNests.empty())
    return PreservedAnalyses::all();

  // Pre-compute which loops have their own work (excluding sub-loops).
  // We record the header block since Loop* pointers become invalid after
  // unrolling, but header blocks persist. This ensures we don't unroll
  // "pass-through" loops that originally had no work of their own, even
  // after their inner loops are unrolled into them.
  SmallPtrSet<BasicBlock *, 16> LoopsWithOwnWork;
  for (Loop *L : VecHintNests)
    collectLoopsWithWork(L, LoopsWithOwnWork);

  // Iteratively unroll small innermost loops within vec-hint nests.
  // After each unroll, re-discover since the loop tree has changed.
  bool AnyChanged = true;
  while (AnyChanged) {
    AnyChanged = false;

    // Re-collect outer loops within vec-hint nests that have unrollable
    // inner loops.
    SmallVector<Loop *, 4> OuterLoops;
    for (Loop *TopLevel : LI) {
      SmallVector<Loop *, 8> Worklist;
      Worklist.push_back(TopLevel);
      while (!Worklist.empty()) {
        Loop *L = Worklist.pop_back_val();
        if (!L->isInnermost() && VecHintNests.count(L)) {
          SmallVector<Loop *, 8> Candidates;
          collectInnermostUnrollCandidates(L, SE, Candidates);
          if (!Candidates.empty())
            OuterLoops.push_back(L);
        }
        for (Loop *Sub : *L)
          Worklist.push_back(Sub);
      }
    }

    if (OuterLoops.empty())
      break;

    LLVM_DEBUG(dbgs() << "LoopUnrollForVec: Found " << OuterLoops.size()
                      << " outer loop(s) with unrollable inner loops in "
                      << F.getName() << "\n");

    for (Loop *OuterL : OuterLoops) {
      SmallVector<Loop *, 8> Candidates;
      collectInnermostUnrollCandidates(OuterL, SE, Candidates);

      if (Candidates.empty())
        continue;

      LLVM_DEBUG(dbgs() << "LoopUnrollForVec: " << Candidates.size()
                        << " inner loop candidate(s) in outer loop at "
                        << OuterL->getHeader()->getName() << "\n");

      for (Loop *InnerL : Candidates) {
        unsigned TripCount = SE.getSmallConstantTripCount(InnerL);
        if (TripCount == 0)
          continue;

        // Skip loops that originally had no work of their own (only contained
        // sub-loops). This avoids unrolling pass-through nesting levels even
        // after their inner loops have been unrolled into them.
        if (!LoopsWithOwnWork.count(InnerL->getHeader())) {
          LLVM_DEBUG(dbgs() << "LoopUnrollForVec: Skipping loop at "
                            << InnerL->getHeader()->getName()
                            << " (no work excluding sub-loops)\n");
          continue;
        }

        unsigned BodySize = estimateLoopBodySize(InnerL);
        unsigned UnrolledSize = BodySize * TripCount;

        if (UnrolledSize > UnrollForVecMaxUnrolledSize) {
          LLVM_DEBUG(dbgs()
                     << "LoopUnrollForVec: Skipping loop at "
                     << InnerL->getHeader()->getName() << " (unrolled size "
                     << UnrolledSize << " exceeds limit "
                     << UnrollForVecMaxUnrolledSize << ")\n");
          continue;
        }

        LLVM_DEBUG(dbgs() << "LoopUnrollForVec: Unrolling loop at "
                          << InnerL->getHeader()->getName() << " (trip count "
                          << TripCount << ", body size " << BodySize << ")\n");

        UnrollLoopOptions ULO;
        ULO.Count = TripCount;
        ULO.Force = true;
        ULO.Runtime = false;
        ULO.AllowExpensiveTripCount = false;
        ULO.UnrollRemainder = false;
        ULO.ForgetAllSCEV = true;
        ULO.SCEVExpansionBudget = 0;

        // Ensure loop simplified and LCSSA form before unrolling -- at this
        // pipeline position, these forms may be broken. We must fix all
        // enclosing loops, not just the immediate parent, because UnrollLoop
        // asserts LCSSA on the outermost enclosing loop.
        simplifyLoop(InnerL, &DT, &LI, &SE, &AC, nullptr, false);
        formLCSSARecursively(*InnerL, DT, &LI, &SE);
        for (Loop *Enclosing = InnerL->getParentLoop(); Enclosing;
             Enclosing = Enclosing->getParentLoop()) {
          simplifyLoop(Enclosing, &DT, &LI, &SE, &AC, nullptr, false);
          formLCSSARecursively(*Enclosing, DT, &LI, &SE);
        }

        LoopUnrollResult Result =
            UnrollLoop(InnerL, ULO, &LI, &SE, &DT, &AC, &TTI, &ORE,
                       /*PreserveLCSSA=*/true);

        if (Result != LoopUnrollResult::Unmodified) {
          Changed = true;
          AnyChanged = true;
          // Loop tree changed -- break out and re-discover.
          break;
        }
      }

      // If we unrolled something, re-discover from scratch.
      if (AnyChanged)
        break;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
