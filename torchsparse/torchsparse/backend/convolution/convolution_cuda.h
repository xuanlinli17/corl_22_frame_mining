#pragma once

#include <torch/torch.h>
#include <torch/serialize/tensor.h>
#include <ATen/ATen.h>

void convolution_forward_cuda(at::Tensor in_feat, at::Tensor out_feat,
                              at::Tensor kernel, at::Tensor neighbor_map,
                              at::Tensor neighbor_offset, const bool transpose);

void convolution_backward_cuda(at::Tensor in_feat, at::Tensor grad_in_feat,
                               at::Tensor grad_out_feat, at::Tensor kernel,
                               at::Tensor grad_kernel, at::Tensor neighbor_map,
                               at::Tensor neighbor_offset,
                               const bool transpose);
