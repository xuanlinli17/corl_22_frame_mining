#pragma once

#include <torch/torch.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>

at::Tensor count_cpu(const at::Tensor idx, const int s);
