module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_extracted_load_or_store_vec_seed(
      %arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>},
      %gm_scalar: memref<?xf32, #hivm.address_space<gm>>,
      %gm_k: memref<16x16xf16>)
      attributes {WorkspaceArgIdx = 0 : i16,
                  func_dyn_memref_args = dense<[true, true, true]> : vector<3xi1>,
                  global_kernel = "local", hacc.entry,
                  hacc.function_kind = #hacc.function_kind<DEVICE>,
                  hivm.func_core_type = #hivm.func_core_type<MIX>,
                  mix_mode = "mix"} {
    %A = tensor.empty() : tensor<16x16xf16>
    %c0 = arith.constant 0 : i32
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %c16_idx = arith.constant 16 : index
    %c2_i32 = arith.constant 2 : i32
    %true = arith.constant true
    %bound = arith.constant 4 : i32
    %ub_scalar_buf = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
    scf.for %i = %c0 to %bound step %c2_i32 : i32 {
      scf.for %j = %c0_idx to %c16_idx step %c1_idx {
        %off = arith.constant 16 : index
        %addr = memref.reinterpret_cast %gm_scalar to offset: [%off], sizes: [1], strides: [1]
            : memref<?xf32, #hivm.address_space<gm>>
              to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %v = memref.load %addr[%c0_idx]
            : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        memref.store %v, %ub_scalar_buf[%j] : memref<16xf32, #hivm.address_space<ub>>
      } {ExtractedLoadOrStore}

      %allocK = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%gm_k : memref<16x16xf16>) outs(%allocK : memref<16x16xf16>)
      %K = bufferization.to_tensor %allocK : memref<16x16xf16>

      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%A, %K, %true, %c16_idx, %c16_idx, %c16_idx
          : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index)
          outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>)
          outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0
          : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %dot_t = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %vdest = tensor.empty() : tensor<16x16xf16>
      %sum = hivm.hir.vadd ins(%dot_t, %dot_t : tensor<16x16xf16>, tensor<16x16xf16>)
          outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%sum : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)
    }
    return
  }
}
