/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>

namespace libkineto {
template <class ClockT>
inline int64_t timeSinceEpoch(const std::chrono::time_point<ClockT>& t) {

#if !defined(KINETO_SCOPE) && defined(LEGACY_PYTORCH)
  // ### This workaround should not be open sourced. ####
  //
  // For PyTorch versions < 2.4.0, timestamps are in microseconds.
  // Scale down to microseconds for compatibility with newer Kineto versions.
  return std::chrono::duration_cast<std::chrono::microseconds>(
#else
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
#endif
             t.time_since_epoch())
      .count();
}

} // namespace libkineto
