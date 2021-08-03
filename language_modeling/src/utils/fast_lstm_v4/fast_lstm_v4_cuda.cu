// Original code from https://github.com/idiap/fast-transformers/blob/master/fast_transformers/causal_product/causal_product_cuda.cu
// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
// Apoorv Vyas <avyas@idiap.ch>
//
// Modified to implement the fast weight LSTM V3 with FWM update rule*.
// v3 = v2 + res. connection from feed-forward part of the pre-act update term.
// Copyright (c) 2021 Kazuki Irie

#include <torch/extension.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <iostream>


typedef torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
  float_accessor;


// sigmoid
__device__ float sgmf(float x) {
    return 1.f / (1.f + expf(-x));
}


// Forward kernel for fast weight LSTM:
// - coupled input-forget gate.
// - no peephole connections.
// - all activations are sigmoid to get positive recurrent queries.
// Equations; for input z_t ... 
__global__ void fast_lstm_v4_forward_kernel(
    const float_accessor inputs_i,  // input gate
    const float_accessor keys_i,
    const float_accessor values_i,
    const float_accessor betas_i,
    const float_accessor inputs_u,  // update candidate
    const float_accessor keys_u,
    const float_accessor values_u,
    const float_accessor betas_u,
    const float_accessor inputs_o,  // output gate
    const float_accessor keys_o,
    const float_accessor values_o,
    const float_accessor betas_o,
    float_accessor states,
    float_accessor cells,
    float_accessor kv_i,
    float_accessor kv_u,
    float_accessor kv_o,
    float_accessor result,
    float_accessor res_del_nmz,
    float_accessor res_cell,
    float_accessor gate_i,
    float_accessor update_u,
    float_accessor gate_o,
    float_accessor v_old_i,
    float_accessor v_old_u,
    float_accessor v_old_o,
    const int N,
    const int H,
    const int L,
    const int E,
    const int M,
    const int E_per_subblock,
    const int subblocks_per_seq,
    const int T,  // block chunk size in time dim.
    const int l_offset  // multiple of T, length offset.
) {
    // Each block takes care of one sequence.
    // blockIdx.x = n * H + h
    int n = blockIdx.x / H;  // batch id
    int h = blockIdx.x % H;  // head id

    // threadIdx.x = e_local*M + m
    // Local e coordinate within E_per_subblock sub-block.
    int e_local = threadIdx.x / M;
    int m = threadIdx.x % M;

    const int E_block = subblocks_per_seq * E_per_subblock;

    // Load the shared memory
    const int shared_kv_size = E_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv_i = shared_mem;
    float* shared_kv_u = shared_kv_i + shared_kv_size;
    float* shared_kv_o = shared_kv_u + shared_kv_size;

    float* shared_states = shared_kv_o + shared_kv_size;
    float* shared_cells = shared_states + M;
    float* shared_gate_i = shared_cells + M;
    float* shared_update = shared_gate_i + M;
    float* shared_gate_o = shared_update + M;

    float* shared_v_old_i = shared_gate_o + M;
    float* shared_v_old_u = shared_v_old_i + M;
    float* shared_v_old_o = shared_v_old_u + M;
    float* shared_betas_i = shared_v_old_o + M;
    float* shared_betas_u = shared_betas_i + T;
    float* shared_betas_o = shared_betas_u + T;
    float* softmax_denom = shared_betas_o + T;
    float* max_value = softmax_denom + 1;

    float* shared_values_i = max_value + 1;  // input gate
    float* shared_keys_i = shared_values_i + M*T;
    float* shared_inputs_i = shared_keys_i + E_block*T;

    float* shared_values_u = shared_inputs_i + M*T;  // update candidate
    float* shared_keys_u = shared_values_u + M*T;
    float* shared_inputs_u = shared_keys_u + E_block*T;

    float* shared_values_o = shared_inputs_u + M*T;  // output gate
    float* shared_keys_o = shared_values_o + M*T;
    float* shared_inputs_o = shared_keys_o + E_block*T;

    const float eps = 1e-6;

    if (threadIdx.x < M) {
        // m = threadIdx.x if threadIdx.x < M.
        // shared_results[m] = 0.f;
        shared_update[m] = 0.f;
        shared_gate_i[m] = 0.f;
        shared_gate_o[m] = 0.f;
        shared_v_old_i[m] = 0.f;
        shared_v_old_u[m] = 0.f;
        shared_v_old_o[m] = 0.f;
    }
    if (threadIdx.x < 1) {
        softmax_denom[0] = 0.f;
        max_value[0] = 0.f;
    }
    // the last segment is shorter.
    int t_end = (T + l_offset) <= L ? T : L - l_offset;

    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int d = i % M;
        shared_values_i[i] = values_i[n][h][t][d];
        shared_inputs_i[i] = inputs_i[n][h][t][d];

        shared_values_u[i] = values_u[n][h][t][d];
        shared_inputs_u[i] = inputs_u[n][h][t][d];

        shared_values_o[i] = values_o[n][h][t][d];
        shared_inputs_o[i] = inputs_o[n][h][t][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_block); i += blockDim.x)
    {
        int t = int(i / E_block) + l_offset;
        int d = (i % E_block);
        if (d < E) {
            shared_keys_i[i] = keys_i[n][h][t][d];
            shared_keys_u[i] = keys_u[n][h][t][d];
            shared_keys_o[i] = keys_o[n][h][t][d];
        }
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        shared_betas_i[i] = betas_i[n][h][t][0];
        shared_betas_u[i] = betas_u[n][h][t][0];
        shared_betas_o[i] = betas_o[n][h][t][0];
    }
    __syncthreads();
    if (n >= N) {
        return;
    }
    int e;
    int kv_idx;
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            shared_kv_i[kv_idx] = kv_i[n][h][e][m];
            shared_kv_u[kv_idx] = kv_u[n][h][e][m];
            shared_kv_o[kv_idx] = kv_o[n][h][e][m];
        }
    }
    // init variables
    if (threadIdx.x < M) {
        // initialize RNN state
        shared_states[m] = states[n][h][0][m];
        shared_cells[m] = cells[n][h][0][m];
    }
    int e_abs;
    float resi, resu, reso;
    float max_val, tmp_max;
    // float res_v_old_i, res_v_old_u, res_v_old_o;
    for (int t=0; t<t_end; t++) {  // loop over time in the segment
        int l = t + l_offset;  // absolute position in time
        int m_abs = t*M + m;

        // For stable softmax
        if (threadIdx.x < 1) {  // Not parallelized! this should be improved!
            max_val = shared_states[0];
            for (int i = 1; i < M; i++) {
                tmp_max = shared_states[i];
                if (tmp_max > max_val) {
                    max_val = tmp_max;
                }
            }
            max_value[0] = max_val;
        }
        __syncthreads();

        // compute denominator for softmax
        if (threadIdx.x < M) {
            shared_states[m] = expf(shared_states[m] - max_value[0]);
            atomicAdd(
                &softmax_denom[0],
                shared_states[m]
            );
        }
        __syncthreads();

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // get old value
                float res_v_old_i = shared_kv_i[kv_idx] * shared_keys_i[e_abs];
                atomicAdd(
                    &shared_v_old_i[m],
                    res_v_old_i
                );
                float res_v_old_u = shared_kv_u[kv_idx] * shared_keys_u[e_abs];
                atomicAdd(
                    &shared_v_old_u[m],
                    res_v_old_u
                );
                float res_v_old_o = shared_kv_o[kv_idx] * shared_keys_o[e_abs];
                atomicAdd(
                    &shared_v_old_o[m],
                    res_v_old_o
                );
            }
        }
        __syncthreads();
        // compute new value to be inserted
        float v_insert_i = shared_betas_i[t] * 
            (shared_values_i[m_abs] - shared_v_old_i[m]);
        float v_insert_u = shared_betas_u[t] * 
            (shared_values_u[m_abs] - shared_v_old_u[m]);
        float v_insert_o = shared_betas_o[t] * 
            (shared_values_o[m_abs] - shared_v_old_o[m]);

        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // Update all fast weights
                shared_kv_i[kv_idx] += shared_keys_i[e_abs] * v_insert_i;
                shared_kv_u[kv_idx] += shared_keys_u[e_abs] * v_insert_u;
                shared_kv_o[kv_idx] += shared_keys_o[e_abs] * v_insert_o;

                float soft_out = shared_states[e] / (softmax_denom[0] + eps);
                // Compute recurrent preactivation terms 
                resi = soft_out * shared_kv_i[kv_idx];
                atomicAdd(
                    &shared_gate_i[m],
                    resi
                );
                resu = soft_out * shared_kv_u[kv_idx];
                atomicAdd(
                    &shared_update[m],
                    resu
                );
                reso = soft_out * shared_kv_o[kv_idx];
                atomicAdd(
                    &shared_gate_o[m],
                    reso
                );
            }
        }
        __syncthreads();
        float out, new_cell;
        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            // Ideally skip this for eval. Saving for bwd pass.
            float tmp = shared_states[m] / (softmax_denom[0] + eps);
            atomicAdd(
                &res_del_nmz[n][h][l][m],
                tmp
            );
            // sigmoid
            shared_gate_i[m] = sgmf(shared_gate_i[m] + shared_inputs_i[m_abs]);
            // FINDME v4
            shared_update[m] = shared_update[m] + shared_inputs_u[m_abs];
            shared_gate_o[m] = sgmf(shared_gate_o[m] + shared_inputs_o[m_abs]);

            new_cell = shared_gate_i[m] * shared_update[m]
              + (1.f - shared_gate_i[m]) * shared_cells[m];
            out = shared_gate_o[m] * new_cell;
            // out = expf(shared_gate_o[m] * new_cell);
            // atomicAdd(
            //     &softmax_denom[0],
            //     out
            // );
            // write back intermediate results to be used for backward pass.
            atomicAdd(
                &result[n][h][l][m],
                out
            );
            shared_states[m] = out;  // state update
            atomicAdd(
                &res_cell[n][h][l][m],
                new_cell
            );
            shared_cells[m] = new_cell;

            float out_i = shared_gate_i[m];
            atomicAdd(
                &gate_i[n][h][l][m],
                out_i
            );
            float out_u = shared_update[m];
            atomicAdd(
                &update_u[n][h][l][m],
                out_u
            );
            float out_o = shared_gate_o[m];
            atomicAdd(
                &gate_o[n][h][l][m],
                out_o
            );
            // initialize gates and update:
            shared_gate_i[m] = 0.f;
            shared_update[m] = 0.f;
            shared_gate_o[m] = 0.f;
            
            float r2i = shared_v_old_i[m];
            atomicAdd(
                &v_old_i[n][h][l][m],
                r2i
            );
            shared_v_old_i[m] = 0.f;

            float r2u = shared_v_old_u[m];
            atomicAdd(
                &v_old_u[n][h][l][m],
                r2u
            );
            shared_v_old_u[m] = 0.f;

            float r2o = shared_v_old_o[m];
            atomicAdd(
                &v_old_o[n][h][l][m],
                r2o
            );
            shared_v_old_o[m] = 0.f;
        }
        __syncthreads();
        if (threadIdx.x < 1) {
            softmax_denom[0] = 0.f;
        }
        __syncthreads();
    }
    __syncthreads();
    // write back to kv to be carried over to the next segment.
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            kv_i[n][h][e][m] = shared_kv_i[kv_idx];
            kv_u[n][h][e][m] = shared_kv_u[kv_idx];
            kv_o[n][h][e][m] = shared_kv_o[kv_idx];
        }
    }
    if (threadIdx.x < M) {
        states[n][h][0][m] = shared_states[m];
        cells[n][h][0][m] = shared_cells[m];
    }
}


// Forward
void fast_lstm_v4_forward(
    const torch::Tensor inputs_i,  // input gate
    const torch::Tensor keys_i,
    const torch::Tensor values_i,
    const torch::Tensor betas_i,
    const torch::Tensor inputs_u,  // update
    const torch::Tensor keys_u,
    const torch::Tensor values_u,
    const torch::Tensor betas_u,
    const torch::Tensor inputs_o,  // output gate
    const torch::Tensor keys_o,
    const torch::Tensor values_o,
    const torch::Tensor betas_o,
    torch::Tensor states,  // init states
    torch::Tensor cells,  // init cell states
    torch::Tensor kv_i,  // might be non zero if carried over from previous seg
    torch::Tensor kv_u,
    torch::Tensor kv_o,
    torch::Tensor outputs,
    torch::Tensor nmz_delay,  // softmax output delayed
    torch::Tensor cell_outs,
    torch::Tensor gate_i,
    torch::Tensor update_u,
    torch::Tensor gate_o,
    torch::Tensor v_old_i,
    torch::Tensor v_old_u,
    torch::Tensor v_old_o
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(queries));
    torch::DeviceGuard _guard(inputs_i.device());
    int N = inputs_i.size(0);
    int H = inputs_i.size(1);
    int L = inputs_i.size(2);
    int E = inputs_i.size(3);
    int M = values_i.size(3);

    // int threads = 1024;
    int threads = 512;  // avoid edge cases.

    // Shared mem max size is 48KB
    int MUL_PER_BLOCK = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MUL_PER_BLOCK = int(MUL_PER_BLOCK / M) *  M;
    threads = MUL_PER_BLOCK;
    const int subblocks_per_seq = ((E*M) + threads -1) / threads;

    const int E_per_subblock = MUL_PER_BLOCK / M;
    const int E_block = subblocks_per_seq * E_per_subblock;
    // int blocks  = N*H*blocks_per_sequence;
    int blocks = N*H;  // total number of sequences
    // 3 fast weight, 2 output/cells, 3 transforms, 3 for v_old,
    // 1 softmax denominator, +1 to store max for stable softmax.
    int shared_mem_const = (E_block * 3 + 5 + 3)*M + 1 + 1;
    // M for value, 2 * E for query and key.
    int shared_mem_per_time = 6*M + 3*E_block + 3;

    // Max shared memory size:
    // 12 * 1024 * 4 (float) = 49152 (48KB)
    // for Turing: 65536 (64KB)
    // for Volta: 98304 (96KB)
    int maxB;
    int device_id = 0;
    // int device_id = inputs_i.device();
    // Should to be faster than `cudaGetDeviceProperties` according to: https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
    cudaDeviceGetAttribute(&maxB,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    // std::cout << "Max shared mem: " << maxB << std::endl;
    int maxF = maxB / sizeof(float);
    // Following is needed for sm > 48KB
    cudaFuncSetAttribute(fast_lstm_v4_forward_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maxB);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    assert(maxF - shared_mem_const > 0 &&
        "`d_head` too large. To obtain large models, keep `d_head` small"
        "e.g. 16 and increase the number of heads instead.");
    // std::cout << "Max shared mem:  " << maxF * sizeof(float) << std::endl;
    // std::cout << "Shared mem const (float): " << 
    //   shared_mem_const * sizeof(float) << std::endl;
    // std::cout << "Remainder: " << maxF - shared_mem_const << std::endl;
    // std::cout << "Shared per time: " << shared_mem_per_time << std::endl;
    const int T = int((maxF - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_forward =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);
    // std::cout << "Total used shared mem: " << shared_mem_forward << std::endl;

    for (int l_offset=0; l_offset < L; l_offset += T) {
     fast_lstm_v4_forward_kernel
            <<<blocks, MUL_PER_BLOCK, shared_mem_forward>>>(
            inputs_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            inputs_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            inputs_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            states.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            cells.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            kv_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            outputs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            nmz_delay.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            cell_outs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            gate_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            update_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            gate_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq, T, l_offset
        );
    }
}


// Backward kernel, output gate
__global__ void fast_lstm_v4_backward_kernel(
    const float_accessor keys_i,
    const float_accessor values_i,
    const float_accessor betas_i,
    const float_accessor keys_u,
    const float_accessor values_u,
    const float_accessor betas_u,
    const float_accessor keys_o,
    const float_accessor values_o,
    const float_accessor betas_o,
    const float_accessor v_out_i,
    const float_accessor v_out_u,
    const float_accessor v_out_o,
    const float_accessor rnn_out,
    const float_accessor rnn_out_delayed,
    const float_accessor cell_out,
    const float_accessor u_minus_c,
    const float_accessor grad_out,
    const float_accessor gate_i,
    const float_accessor update_u,
    const float_accessor gate_o,
    float_accessor grad_h,  // output tmp grad
    float_accessor grad_c,  // cell tmp grad
    float_accessor kv_i,  // kv memory from the forward pass
    float_accessor kv_u,
    float_accessor kv_o,
    float_accessor grad_kv_i,  // kv temporal grad
    float_accessor grad_kv_u,
    float_accessor grad_kv_o,
    float_accessor grad_inputs_i,  // input gate
    float_accessor grad_keys_i,
    float_accessor grad_values_i,
    float_accessor grad_betas_i,
    float_accessor grad_inputs_u,  // update
    float_accessor grad_keys_u,
    float_accessor grad_values_u,
    float_accessor grad_betas_u,
    float_accessor grad_inputs_o,  // output gate
    float_accessor grad_keys_o,
    float_accessor grad_values_o,
    float_accessor grad_betas_o,
    int N,
    int H,
    int L,
    int E,
    int M,
    int E_per_subblock,
    int subblocks_per_seq,
    int T,
    int l_offset
) {
    // Each block takes care of one sequence.
    // blockIdx.x = n * H + h
    int n = blockIdx.x / H;
    int h = blockIdx.x % H;

    // threadIdx.x = e_local*M + m
    // Local e coordinate within E_per_subblock sub-block.
    int e_local = threadIdx.x / M;
    int m = threadIdx.x % M;

    const int E_block = subblocks_per_seq * E_per_subblock;

    // Load the shared memory for KV
    const int shared_kv_size = E_block * M;
    extern __shared__ float shared_mem[];
    float* shared_kv_i = shared_mem;
    float* shared_grad_kv_i = shared_mem + shared_kv_size;
    float* shared_kv_u = shared_grad_kv_i + shared_kv_size;
    float* shared_grad_kv_u = shared_kv_u + shared_kv_size;
    float* shared_kv_o = shared_grad_kv_u + shared_kv_size;
    float* shared_grad_kv_o = shared_kv_o + shared_kv_size;

    float* shared_res_zi = shared_grad_kv_o + shared_kv_size;
    float* shared_res_zu = shared_res_zi + M;
    float* shared_res_zo = shared_res_zu + M;

    float* shared_res_k_i = shared_res_zo + M;
    float* shared_res_k_u = shared_res_k_i + M;
    float* shared_res_k_o = shared_res_k_u + M;

    float* shared_res_v_i = shared_res_k_o + M;
    float* shared_res_v_u = shared_res_v_i + M;
    float* shared_res_v_o = shared_res_v_u + M;

    float* shared_grad_v_old_i = shared_res_v_o + M;
    float* shared_grad_v_old_u = shared_grad_v_old_i + M;
    float* shared_grad_v_old_o = shared_grad_v_old_u + M;

    float* shared_res_beta_i = shared_grad_v_old_o + M;
    float* shared_res_beta_u = shared_res_beta_i + 1;
    float* shared_res_beta_o = shared_res_beta_u + 1;

    float* grad_sft_cst = shared_res_beta_o + 1;

    float* shared_gradout = grad_sft_cst + 1;

    float* shared_keys_i = shared_gradout + M*T;
    float* shared_values_i = shared_keys_i + E_block*T;

    float* shared_keys_u = shared_values_i + M*T;
    float* shared_values_u = shared_keys_u + E_block*T;

    float* shared_keys_o = shared_values_u + M*T;
    float* shared_values_o = shared_keys_o + E_block*T;

    float* shared_rnn_out = shared_values_o + M*T;
    float* shared_rnn_out_delayed = shared_rnn_out + M*T;

    float* shared_c = shared_rnn_out_delayed + M*T;
    float* shared_u_m_c = shared_c + M*T;

    float* shared_gate_i = shared_u_m_c + M*T;
    float* shared_update = shared_gate_i + M*T;
    float* shared_gate_o = shared_update + M*T;

    float* shared_v_old_i = shared_gate_o + M*T;
    float* shared_v_old_u = shared_v_old_i + M*T;
    float* shared_v_old_o = shared_v_old_u + M*T;

    float* shared_betas_i = shared_v_old_o + M*T;
    float* shared_betas_u = shared_betas_i + T;
    float* shared_betas_o = shared_betas_u + T;

    float* shared_grad_h = shared_betas_o + T;
    float* shared_grad_c = shared_grad_h + M*T;

    if (threadIdx.x < M) {
        shared_res_zi[m] = 0.f;
        shared_res_zu[m] = 0.f;
        shared_res_zo[m] = 0.f;

        shared_res_k_i[m] = 0.f;
        shared_res_k_u[m] = 0.f;
        shared_res_k_o[m] = 0.f;

        shared_res_v_i[m] = 0.f;
        shared_res_v_u[m] = 0.f;
        shared_res_v_o[m] = 0.f;

        shared_grad_v_old_i[m] = 0.f;
        shared_grad_v_old_u[m] = 0.f;
        shared_grad_v_old_o[m] = 0.f;
    }
    if (threadIdx.x < 1) {
        shared_res_beta_i[0] = 0.f;
        shared_res_beta_u[0] = 0.f;
        shared_res_beta_o[0] = 0.f;
        grad_sft_cst[0] = 0.f;  // offset for grad softmax
    }
    // Everythig goes backward
    int t_end = (T + l_offset) <= L ? T : L - l_offset;
    for (int i = threadIdx.x; i < (t_end*M); i += blockDim.x)
    {
        int t = int(i / M) + l_offset;
        int t_bw = L - 1 - t;
        int d = i % M;
        shared_gradout[i] = grad_out[n][h][t_bw][d];

        shared_rnn_out[i] = rnn_out[n][h][t_bw][d];
        shared_c[i] = cell_out[n][h][t_bw][d];
        shared_u_m_c[i] = u_minus_c[n][h][t_bw][d];

        shared_values_i[i] = values_i[n][h][t_bw][d];
        shared_values_u[i] = values_u[n][h][t_bw][d];
        shared_values_o[i] = values_o[n][h][t_bw][d];

        shared_v_old_i[i] = v_out_i[n][h][t_bw][d];
        shared_v_old_u[i] = v_out_u[n][h][t_bw][d];
        shared_v_old_o[i] = v_out_o[n][h][t_bw][d];

        shared_gate_i[i] = gate_i[n][h][t_bw][d];
        shared_update[i] = update_u[n][h][t_bw][d];
        shared_gate_o[i] = gate_o[n][h][t_bw][d];
    }
    for (int i = threadIdx.x; i < (t_end*E_block); i += blockDim.x)
    {
        int t = int(i / E_block) + l_offset;
        int t_bw = L - 1 - t;
        int d = (i % E_block);
        if (d < E) {
            shared_rnn_out_delayed[i] = rnn_out_delayed[n][h][t_bw][d];
            shared_keys_i[i] = keys_i[n][h][t_bw][d];
            shared_keys_u[i] = keys_u[n][h][t_bw][d];
            shared_keys_o[i] = keys_o[n][h][t_bw][d];
        }
    }
    for (int i = threadIdx.x; i < t_end; i += blockDim.x)
    {
        int t = i + l_offset;
        int t_bw = L - 1 - t;
        shared_betas_i[i] = betas_i[n][h][t_bw][0];
        shared_betas_u[i] = betas_u[n][h][t_bw][0];
        shared_betas_o[i] = betas_o[n][h][t_bw][0];
    }
    __syncthreads();
    if (n >= N) {
        return;
    }
    int e;
    int e_abs;  // absolute idx from t=0
    int kv_idx;
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            shared_kv_i[kv_idx] = kv_i[n][h][e][m];
            shared_grad_kv_i[kv_idx] = grad_kv_i[n][h][e][m];
            shared_kv_u[kv_idx] = kv_u[n][h][e][m];
            shared_grad_kv_u[kv_idx] = grad_kv_u[n][h][e][m];
            shared_kv_o[kv_idx] = kv_o[n][h][e][m];
            shared_grad_kv_o[kv_idx] = grad_kv_o[n][h][e][m];
        }
    }
    if (threadIdx.x < M) {
        // threadIdx.x = m if threadIdx.x < M
        shared_grad_h[m] = grad_h[n][h][0][m];
        shared_grad_c[m] = grad_c[n][h][0][m];
    }

    for (int t=0; t<t_end; t++) {
        int l = t + l_offset;
        int l_b = L - l -1;
        int m_abs = t*M + m;

        if (threadIdx.x < M) {  // element-wise ops only here
            // threadIdx.x = m if threadIdx.x < M
            shared_grad_h[m] += shared_gradout[m_abs];
            // float grad_soft_input =
            //   shared_rnn_out[m_abs] * (shared_grad_h[m] - grad_sft_cst[0]);
            // for output gate
            float grad_o = shared_c[m_abs] * shared_grad_h[m];
            shared_res_zo[m] =
              grad_o * (1.f - shared_gate_o[m_abs]) * shared_gate_o[m_abs];
            // grad c, no sigmoid
            shared_grad_c[m] += shared_gate_o[m_abs] * shared_grad_h[m];
            // shared_grad_c[m] += shared_gate_o[m_abs] * shared_grad_h[m]
            //   * sgmf(shared_c[m_abs]) * (1.f - sgmf(shared_c[m_abs]));
            shared_grad_h[m] = 0.f;  // prepare grad h for the next step.
        }
        __syncthreads();  // important to sync

        float v_diff_i = shared_values_i[m_abs] - shared_v_old_i[m_abs];
        float v_ins_i = v_diff_i * shared_betas_i[t];
        
        float v_diff_u = shared_values_u[m_abs] - shared_v_old_u[m_abs];
        float v_ins_u = v_diff_u * shared_betas_u[t];

        float v_diff_o = shared_values_o[m_abs] - shared_v_old_o[m_abs];
        float v_ins_o = v_diff_o * shared_betas_o[t];

        // Output gate
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // grad rec weight
                shared_grad_kv_o[kv_idx] +=
                  shared_res_zo[m] * shared_rnn_out_delayed[e_abs];

                // grad v
                float res_v_o = shared_grad_kv_o[kv_idx] * shared_keys_o[e_abs]
                  * shared_betas_o[t];
                atomicAdd(
                    &shared_res_v_o[m],
                    res_v_o
                );

                // grad k part 1 and 2
                float res_k_o = shared_grad_kv_o[kv_idx] * v_ins_o;
                atomicAdd(
                    &shared_res_k_o[e],
                    res_k_o
                );

                // grad beta
                float res_b_o = shared_grad_kv_o[kv_idx] * shared_keys_o[e_abs]
                  * v_diff_o;
                atomicAdd(
                    &shared_res_beta_o[0],
                    res_b_o
                );

                // pass grad for the next time step.
                float res_h_o = shared_res_zo[m] * shared_kv_o[kv_idx];
                atomicAdd(
                    &shared_grad_h[e],
                    res_h_o
                );  // contribution from output gate
            }
        }
        __syncthreads();

        if (threadIdx.x < M) {
            // grad input gate
            float grad_i = shared_grad_c[m] * shared_u_m_c[m_abs];
            shared_res_zi[m] = 
              grad_i * (1.f - shared_gate_i[m_abs]) * shared_gate_i[m_abs];

            // grad update
            shared_res_zu[m] = shared_grad_c[m] * shared_gate_i[m_abs];

            // prepare grad c for the next time step
            shared_grad_c[m] = shared_grad_c[m] * (1.f - shared_gate_i[m_abs]);
        }
        __syncthreads();  // important to sync

        // Grad for input gate and update transformation
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // grad rec weight
                shared_grad_kv_i[kv_idx] +=
                  shared_res_zi[m] * shared_rnn_out_delayed[e_abs];

                shared_grad_kv_u[kv_idx] +=
                  shared_res_zu[m] * shared_rnn_out_delayed[e_abs];

                // grad v
                float res_v_i = shared_grad_kv_i[kv_idx] * shared_keys_i[e_abs]
                  * shared_betas_i[t];
                atomicAdd(
                    &shared_res_v_i[m],
                    res_v_i
                );
                float res_v_u = shared_grad_kv_u[kv_idx] * shared_keys_u[e_abs]
                  * shared_betas_u[t];
                atomicAdd(
                    &shared_res_v_u[m],
                    res_v_u
                );

                // grad k
                float res_k_i = 
                  shared_grad_kv_i[kv_idx] * v_ins_i;
                atomicAdd(
                    &shared_res_k_i[e],
                    res_k_i
                );
                float res_k_u = 
                  shared_grad_kv_u[kv_idx] * v_ins_u;
                atomicAdd(
                    &shared_res_k_u[e],
                    res_k_u
                );

                // grad beta
                float res_b_i = shared_grad_kv_i[kv_idx] * shared_keys_i[e_abs]
                  * v_diff_i;
                atomicAdd(
                    &shared_res_beta_i[0],
                    res_b_i
                );
                float res_b_u = shared_grad_kv_u[kv_idx] * shared_keys_u[e_abs]
                  * v_diff_u;
                atomicAdd(
                    &shared_res_beta_u[0],
                    res_b_u
                );

                // pass gradients to the next time step
                float res_h_i = shared_res_zi[m] * shared_kv_i[kv_idx];
                atomicAdd(
                    &shared_grad_h[e],
                    res_h_i
                );  // contribution from input gate
                float res_h_u = shared_res_zu[m] * shared_kv_u[kv_idx];
                atomicAdd(
                    &shared_grad_h[e],
                    res_h_u
                );  // contribution from update transformation
            }
        }
        __syncthreads();
        // compute constant for grad softmax
        if (threadIdx.x < M) {
            float cst = shared_grad_h[m] * shared_rnn_out_delayed[m_abs];
            atomicAdd(
                &grad_sft_cst[0],
                cst
            );
        }
        __syncthreads();
        if (threadIdx.x < M) {
            shared_grad_h[m] = shared_rnn_out_delayed[m_abs]
              * (shared_grad_h[m] - grad_sft_cst[0]);
        }
        if (threadIdx.x < 1) {
            grad_sft_cst[0] = 0.f;
        }
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // reverse update fast weight
                shared_kv_i[kv_idx] -= shared_keys_i[e_abs] * v_ins_i;
                shared_kv_u[kv_idx] -= shared_keys_u[e_abs] * v_ins_u;
                shared_kv_o[kv_idx] -= shared_keys_o[e_abs] * v_ins_o;

                // grad v_old
                float res_v_old_i = - (shared_grad_kv_i[kv_idx]
                    * shared_betas_i[t] * shared_keys_i[e_abs]);
                  atomicAdd(
                    &shared_grad_v_old_i[m],
                    res_v_old_i
                  );
                float res_v_old_u = - (shared_grad_kv_u[kv_idx]
                    * shared_betas_u[t] * shared_keys_u[e_abs]);
                  atomicAdd(
                    &shared_grad_v_old_u[m],
                    res_v_old_u
                  );
                float res_v_old_o = - (shared_grad_kv_o[kv_idx]
                    * shared_betas_o[t] * shared_keys_o[e_abs]);
                  atomicAdd(
                    &shared_grad_v_old_o[m],
                    res_v_old_o
                  );
            }
        }
        __syncthreads();
        // remaining key grad
        for (int sub=0; sub<subblocks_per_seq; sub++) {
            e = sub * E_per_subblock + e_local;
            e_abs = t*E_block + e;
            kv_idx = threadIdx.x + sub * blockDim.x;
            if (e < E) {
                // Input gate
                float res_kp3_i = shared_grad_v_old_i[m] * shared_kv_i[kv_idx];
                atomicAdd(
                    &shared_res_k_i[e],
                    res_kp3_i
                );  // remaining key grad
                // grad kv via v old
                shared_grad_kv_i[kv_idx] +=
                  shared_grad_v_old_i[m] * shared_keys_i[e_abs];
                // Update transform
                float res_kp3_u = shared_grad_v_old_u[m] * shared_kv_u[kv_idx];
                atomicAdd(
                    &shared_res_k_u[e],
                    res_kp3_u
                );  // remaining key grad
                // grad kv via v old
                shared_grad_kv_u[kv_idx] +=
                  shared_grad_v_old_u[m] * shared_keys_u[e_abs];
                // Output gate
                float res_kp3_o = shared_grad_v_old_o[m] * shared_kv_o[kv_idx];
                atomicAdd(
                    &shared_res_k_o[e],
                    res_kp3_o
                );  // remaining key grad
                // grad kv via v old
                shared_grad_kv_o[kv_idx] +=
                  shared_grad_v_old_o[m] * shared_keys_o[e_abs];
            }
        }
        __syncthreads();

        if (threadIdx.x < M) {
            // m = threadIdx.x if threadIdx.x < M
            // feed-forward part
            float rzi = shared_res_zi[m];
            atomicAdd(
                &grad_inputs_i[n][h][l_b][m],
                rzi
            );
            float rzu = shared_res_zu[m];
            atomicAdd(
                &grad_inputs_u[n][h][l_b][m],
                rzu
            );
            float rzo = shared_res_zo[m];
            atomicAdd(
                &grad_inputs_o[n][h][l_b][m],
                rzo
            );
            // keys 
            float rki = shared_res_k_i[m];
            atomicAdd(
                &grad_keys_i[n][h][l_b][m],
                rki
            );
            float rku = shared_res_k_u[m];
            atomicAdd(
                &grad_keys_u[n][h][l_b][m],
                rku
            );
            float rko = shared_res_k_o[m];
            atomicAdd(
                &grad_keys_o[n][h][l_b][m],
                rko
            );
            // values
            float rvi = shared_res_v_i[m];
            atomicAdd(
                &grad_values_i[n][h][l_b][m],
                rvi
            );
            float rvu = shared_res_v_u[m];
            atomicAdd(
                &grad_values_u[n][h][l_b][m],
                rvu
            );
            float rvo = shared_res_v_o[m];
            atomicAdd(
                &grad_values_o[n][h][l_b][m],
                rvo
            );
            // reset 
            shared_res_k_i[m] = 0.f;
            shared_res_k_u[m] = 0.f;
            shared_res_k_o[m] = 0.f;  
            
            shared_res_v_i[m] = 0.f;
            shared_res_v_u[m] = 0.f;
            shared_res_v_o[m] = 0.f;       
            
            shared_grad_v_old_i[m] = 0.f;
            shared_grad_v_old_u[m] = 0.f;
            shared_grad_v_old_o[m] = 0.f;
        }
        __syncthreads();
        if (threadIdx.x < 1) {
            // input
            atomicAdd(
                &grad_betas_i[n][h][l_b][0],
                shared_res_beta_i[0]
            );
            shared_res_beta_i[0] = 0.f;
            // update
            atomicAdd(
                &grad_betas_u[n][h][l_b][0],
                shared_res_beta_u[0]
            );
            shared_res_beta_u[0] = 0.f;
            // output gate
            atomicAdd(
                &grad_betas_o[n][h][l_b][0],
                shared_res_beta_o[0]
            );
            shared_res_beta_o[0] = 0.f;
        }
        __syncthreads();
    }
    __syncthreads();
    // write back temporal gradients.
    for (int sub=0; sub<subblocks_per_seq; sub++) {
        e = sub * E_per_subblock + e_local;
        kv_idx = threadIdx.x + sub * blockDim.x;
        if (e < E) {
            kv_i[n][h][e][m] = shared_kv_i[kv_idx];
            grad_kv_i[n][h][e][m] = shared_grad_kv_i[kv_idx];

            kv_u[n][h][e][m] = shared_kv_u[kv_idx];
            grad_kv_u[n][h][e][m] = shared_grad_kv_u[kv_idx];

            kv_o[n][h][e][m] = shared_kv_o[kv_idx];
            grad_kv_o[n][h][e][m] = shared_grad_kv_o[kv_idx];
        }
    }
    if (threadIdx.x < M) {
        // threadIdx.x = m if threadIdx.x < M
        grad_h[n][h][0][m] = shared_grad_h[m];
        grad_c[n][h][0][m] = shared_grad_c[m];
    }
}


// Backward pass
// This is very shared_mem intensive for the standard LSTM...
void fast_lstm_v4_backward(
    const torch::Tensor grad_out,
    const torch::Tensor keys_i,
    const torch::Tensor values_i,
    const torch::Tensor betas_i,
    const torch::Tensor keys_u,
    const torch::Tensor values_u,
    const torch::Tensor betas_u,
    const torch::Tensor keys_o,
    const torch::Tensor values_o,
    const torch::Tensor betas_o,
    const torch::Tensor v_old_i,
    const torch::Tensor v_old_u,
    const torch::Tensor v_old_o,
    const torch::Tensor outputs,
    const torch::Tensor o_delayed,
    const torch::Tensor cell_out,
    const torch::Tensor u_minus_c,
    const torch::Tensor gate_i,
    const torch::Tensor update_u,
    const torch::Tensor gate_o,
    torch::Tensor fw_mem_i,  // from the forward pass.
    torch::Tensor fw_mem_u,
    torch::Tensor fw_mem_o,
    torch::Tensor grad_in_i,  // input gate
    torch::Tensor grad_ki,
    torch::Tensor grad_vi,
    torch::Tensor grad_bi,
    torch::Tensor grad_in_u,  // update
    torch::Tensor grad_ku,
    torch::Tensor grad_vu,
    torch::Tensor grad_bu,
    torch::Tensor grad_in_o,  // output gate
    torch::Tensor grad_ko,
    torch::Tensor grad_vo,
    torch::Tensor grad_bo
) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_queries));
    torch::DeviceGuard _guard(grad_out.device());

    int N = keys_i.size(0);
    int H = keys_i.size(1);
    int L = keys_i.size(2);
    int E = keys_i.size(3);
    int M = values_i.size(3);

    auto grad_kv_i = torch::zeros({N, H, E, M}, keys_i.options());
    auto grad_kv_u = torch::zeros({N, H, E, M}, keys_i.options());
    auto grad_kv_o = torch::zeros({N, H, E, M}, keys_i.options());

    auto grad_h = torch::zeros({N, H, 1, M}, keys_i.options());
    auto grad_c = torch::zeros({N, H, 1, M}, keys_i.options());

    // const int threads = 1024;
    const int threads = 512;  // avoid edge cases.

    // Gradient output gate ====================================
    int MPB = min(threads, E*M);
    // make sure that MUL_PER_BLOCK is divisible by M;
    MPB = int(MPB / M) *  M;
    const int subblocks_per_seq_value = ((E*M) + MPB - 1)/ MPB;
    const int E_per_subblock = MPB / M;
    const int blocks_value = N*H;
    const int E_block = E_per_subblock * subblocks_per_seq_value;

    // see kernel
    int shared_mem_const = (6 * E_block + 9 + 3)*M + 4;
    int shared_mem_per_time = (12 + 3) * M + 3 * E_block + 3;

    // Max shared memory size:
    // 12 * 1024 * 4 (float) = 49152 (48KB)
    // for Turing: 65536 (64KB)
    // for Volta: 98304 (96KB)
    int maxB;
    int device_id = 0;  // assume all devices to be the same type as device 0.
    // int device_id = keys_i.device();
    // Should to be faster than `cudaGetDeviceProperties` according to: https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/
    cudaDeviceGetAttribute(&maxB,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id);
    // std::cout << "Max shared mem: " << maxB << std::endl;
    int maxF = maxB / sizeof(float);
    // Following is needed for sm > 48KB
    cudaFuncSetAttribute(fast_lstm_v4_backward_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maxB);
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    assert(maxF - shared_mem_const > 0 &&
        "`d_head` too large. To obtain large models, keep `d_head` small"
        "e.g. 16 and increase the number of heads instead.");
    // std::cout << "Max shared mem:  " << maxF * sizeof(float) << std::endl;
    // std::cout << "Shared mem const (float): " << 
    //   shared_mem_const * sizeof(float) << std::endl;
    // std::cout << "Remainder: " << maxF - shared_mem_const << std::endl;
    // std::cout << "Shared per time: " << shared_mem_per_time << std::endl;
    const int T = int((maxF - shared_mem_const) / shared_mem_per_time);
    const int shared_mem_backward =
      ((T*shared_mem_per_time) + shared_mem_const) * sizeof(float);

    for (int l_offset=0; l_offset < L; l_offset += T) {
        fast_lstm_v4_backward_kernel
            <<<blocks_value, MPB, shared_mem_backward>>>(
            keys_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            keys_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            values_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            betas_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            v_old_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            outputs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            o_delayed.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            cell_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            u_minus_c.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            gate_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            update_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            gate_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_h.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_c.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            fw_mem_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            fw_mem_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            fw_mem_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_kv_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_kv_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_kv_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_in_i.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_ki.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_vi.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_bi.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_in_u.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_ku.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_vu.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_bu.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_in_o.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_ko.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_vo.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad_bo.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            N, H, L, E, M, E_per_subblock, subblocks_per_seq_value, T, l_offset
        );
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fast_lstm_v4_forward",
        &fast_lstm_v4_forward,
        "Compute the weighted sum of values but attending only to previous "
        "values."
    );
    m.def(
        "fast_lstm_v4_backward",
        &fast_lstm_v4_backward,
        "Compute the gradients for the fast weight memory."
    );
}
