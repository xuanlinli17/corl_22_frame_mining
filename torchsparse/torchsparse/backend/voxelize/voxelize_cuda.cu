#include <stdio.h>
#include <stdlib.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

#include <THC/THCAtomics.cuh>
#include <cmath>


#define THREADS_PER_BLOCK 256
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// hashing
// input N*F float tensor, pointer to output N'*F int64 tensor, N*1 count
// tensor, N*1 index tensor
template <typename scalar_t>
__global__ void voxelize_forward_kernel(int N, int c, int s,
                                        const scalar_t *__restrict__ data,
                                        const int *__restrict__ idx,
                                        const int *__restrict__ counts,
                                        scalar_t *__restrict__ out) {
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < N && j < c) {
    int pos = idx[i];
    if (pos < 0 || pos >= s || counts[pos] == 0) return;
    atomicAdd(&out[pos * c + j], data[i * c + j] / float(counts[pos]));
  }
}

template <typename scalar_t>
__global__ void voxelize_backward_kernel(int N, int c, int s,
                                         const scalar_t *__restrict__ top_grad,
                                         const int *__restrict__ idx,
                                         const int *__restrict__ counts,
                                         scalar_t *__restrict__ bottom_grad) {
  int i = blockIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (i < N && j < c) {
    int pos = idx[i];
    if (pos < 0 || pos >= s || counts[pos] == 0) return;
    atomicAdd(&bottom_grad[i * c + j],
              top_grad[pos * c + j] / float(counts[pos]));
  }
}

at::Tensor voxelize_forward_cuda(const at::Tensor inputs, const at::Tensor idx,
                                 const at::Tensor counts) {
  int N = inputs.size(0);
  int c = inputs.size(1);
  int N1 = counts.size(0);

  dim3 blocks(N, DIVUP(c, THREADS_PER_BLOCK));
  int max_mum_thread = c <= THREADS_PER_BLOCK ? c : THREADS_PER_BLOCK;
  dim3 threads(1, max_mum_thread);

  at::Tensor out =
      torch::zeros({N1, c}, at::device(idx.device()).dtype(inputs.dtype()));
  
  // printf("%d %d %d\n", THREADS_PER_BLOCK, DIVUP(c, THREADS_PER_BLOCK), N); 
  // return out;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      inputs.type(), "voxelize_forward_cuda", ([&] {
        voxelize_forward_kernel<scalar_t><<<blocks, threads>>>(
            N, c, N1, inputs.data_ptr<scalar_t>(), idx.data_ptr<int>(),
            counts.data_ptr<int>(), out.data_ptr<scalar_t>());
      }));

  return out;
}

at::Tensor voxelize_backward_cuda(const at::Tensor top_grad,
                                  const at::Tensor idx, const at::Tensor counts,
                                  const int N) {
  int c = top_grad.size(1);
  int N1 = counts.size(0);
  
  dim3 blocks(N, DIVUP(c, THREADS_PER_BLOCK));
  int max_mum_thread = c <= THREADS_PER_BLOCK ? c : THREADS_PER_BLOCK;
  dim3 threads(1, max_mum_thread);

  at::Tensor bottom_grad =
      torch::zeros({N, c}, at::device(idx.device()).dtype(top_grad.dtype()));

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "voxelize_backward_cuda", ([&] {
        voxelize_backward_kernel<scalar_t><<<blocks, threads>>>(
            N, c, N1, top_grad.data_ptr<scalar_t>(), idx.data_ptr<int>(),
            counts.data_ptr<int>(), bottom_grad.data_ptr<scalar_t>());
      }));

  return bottom_grad;
}
