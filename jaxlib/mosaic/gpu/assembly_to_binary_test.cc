/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/mosaic/gpu/assembly_to_binary.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mosaic::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::Field;

TEST(PtxasNotesTest, EachLossNoteOfAReportIsItsCodeAndMessageInOrder) {
  // ptxas's verbose report of one kernel: its resource lines, a note of no
  // loss, and two of a loss.
  constexpr char kLog[] =
      "ptxas info    : 0 bytes gmem\n"
      "ptxas info    : Compiling entry function 'kernel' for 'sm_90a'\n"
      "ptxas warning : (C7515) Potential Performance Loss: wgmma.mma_async "
      "instructions are serialized due to the presence of Extern calls in the "
      "function 'kernel'.\n"
      "ptxas info    : Used 168 registers, used 1 barriers\n"
      "ptxas warning : (C7507) Potential Performance Loss: setmaxnreg ignored; "
      "unable to determine register count at entry.  \n"
      "ptxas warning : Unresolved extern function 'vprintf'\n";
  EXPECT_THAT(
      PtxasNotes(kLog),
      ElementsAre(
          testing::AllOf(Field(&PtxasNote::code, "C7515"),
                Field(&PtxasNote::message,
                      "wgmma.mma_async instructions are serialized due to "
                      "the presence of Extern calls in the function "
                      "'kernel'.")),
          testing::AllOf(Field(&PtxasNote::code, "C7507"),
                Field(&PtxasNote::message,
                      "setmaxnreg ignored; unable to determine register "
                      "count at entry."))));
}

TEST(PtxasNotesTest, AReportNotingNoLossHasNone) {
  EXPECT_THAT(PtxasNotes("ptxas info    : Used 32 registers\n"),
              testing::IsEmpty());
}

TEST(LaunchSpilledBytesTest, AnEntrysSpillsAreItsFunctionsStoresAndLoads) {
  // The entry and a device function it calls spilled; a function holding no
  // stats spilled nothing.
  ModuleStats stats;
  stats["kernel"] = KernelStats{192, 136};
  stats["__internal_reduction"] = KernelStats{8, 8};
  EXPECT_EQ(LaunchSpilledBytes(stats, /*entries=*/1), 344);
}

TEST(LaunchSpilledBytesTest, AnEntrySpillingNothingSpilledNoBytes) {
  EXPECT_EQ(LaunchSpilledBytes(ModuleStats(), /*entries=*/1), 0);
}

TEST(LaunchSpilledBytesTest, ABinaryOfSeveralEntriesStatesNone) {
  ModuleStats stats;
  stats["unlaunched"] = KernelStats{64, 64};
  EXPECT_EQ(LaunchSpilledBytes(stats, /*entries=*/2), std::nullopt);
}

}  // namespace
}  // namespace mosaic::gpu
