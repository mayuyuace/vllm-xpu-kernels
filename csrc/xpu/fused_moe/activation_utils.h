#pragma once

enum class ActivationType {
  NONE = -1,
  SILU = 0,
  GELU = 1,
  GELU_TANH = 2,
  SWIGLUOAI = 3,
  RELU2_NO_MUL = 4,
  SWIGLUSTEP = 5,
};
