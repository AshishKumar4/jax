// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-test-pipeline-lower-loop -canonicalize | FileCheck %s
// Run with XLA d867969800d4's Triton b575c0c38d42 plus the fork's patch series.
// Adapted from Triton PR 12738 (6963de9bb738), late shared reader case.
// This checks the typed allocation and its forwarded readers, not compiler logs.
// The released carry computes min(1, 2) + 1 = 2 buffers; stage-2 WGMMA needs 3.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#transposed = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @late_shared_reader
  // CHECK: %[[BUF:.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xbf16
  // CHECK: scf.for
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: %[[VIEW:.*]] = ttg.memdesc_index %[[BUF]]{{.*}}loop.stage = 1
  // CHECK: ttg.local_load %[[VIEW]]
  // CHECK: %[[TRANS:.*]] = ttg.memdesc_trans %[[VIEW]]
  // CHECK: ttng.warp_group_dot %{{.*}}, %[[TRANS]], %{{.*}} {{.*}}loop.stage = 2
  tt.func public @late_shared_reader(%lb: index, %ub: index, %step: index,
      %ptr: tensor<64x64x!tt.ptr<bf16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[1, 16]> : tensor<2xi32>},
      %lhs: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %i = %lb to %ub step %step iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) : index {
      %x = tt.load %ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x64x!tt.ptr<bf16>, #blocked>
      "use"(%x) {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<64x64xbf16, #blocked>) -> ()
      %sh = ttg.local_alloc %x {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem>
      %trans = ttg.memdesc_trans %sh {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared, #smem> -> !ttg.memdesc<64x64xbf16, #transposed, #smem>
      %next = ttng.warp_group_dot %lhs, %trans, %acc {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xbf16, #transposed, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
