#pragma once

#include <torch/torch.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>

at::Tensor hash_cuda(const at::Tensor idx);

at::Tensor kernel_hash_cuda(const at::Tensor idx,
                            const at::Tensor kernel_offset);
