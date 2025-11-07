/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "GenericTraceActivity.h"
#include "output_base.h"

namespace libkineto {
void GenericTraceActivity::log(ActivityLogger& logger) const {
#if defined(LEGACY_PYTORCH)
  // ### This workaround should not be open sourced. ####
  //
  // For PyTorch versions < 2.4.0, timestamps are in microseconds.
  // Scale them to nanoseconds for compatibility with newer Kineto versions.
  if ((activityType == ActivityType::CPU_OP) || (activityType == ActivityType::CPU_INSTANT_EVENT) 
      || (activityType == ActivityType::USER_ANNOTATION) || (activityType == ActivityType::PYTHON_FUNCTION)) {
    GenericTraceActivity newtrace = *this;
    newtrace.startTime *= 1000;
    newtrace.endTime *= 1000;
    logger.handleGenericActivity(newtrace);
    return;
  }
#endif
  logger.handleGenericActivity(*this);
}
} // namespace libkineto
