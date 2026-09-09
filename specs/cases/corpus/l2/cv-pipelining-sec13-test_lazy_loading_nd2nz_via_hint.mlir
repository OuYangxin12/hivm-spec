module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_lazy_loading_nd2nz_via_hint(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %gm_dst: memref<16x16xf16>, %k_src: memref<16x16xf16>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true, true]> : vector<2xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %A = tensor.empty() : tensor<16x16xf16>
    // harvested-away: k_src 提升为函数参数
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = arith.constant 4 : i32
    scf.for %i = %c0 to %bound step %step : i32 {
      %allocK = memref.alloc() : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.nd2nz {dst_continuous} ins(%k_src : memref<16x16xf16>) outs(%allocK : memref<1x1x16x16xf16, #hivm.address_space<cbuf>>)
      %allocK_cast = memref.memory_space_cast %allocK : memref<1x1x16x16xf16, #hivm.address_space<cbuf>> to memref<1x1x16x16xf16>
      %tensorK = bufferization.to_tensor %allocK_cast : memref<1x1x16x16xf16>
      annotation.mark %tensorK {cv_pipeline_lazy_load = true} : tensor<1x1x16x16xf16>

      %dest1 = tensor.empty() : tensor<16x16xf16>
      %dot1 = hivm.hir.mmadL1 ins(%A, %tensorK, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dest1 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws_alloc = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws_cast = memref.memory_space_cast %ws_alloc : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      %ws_t = bufferization.to_tensor %ws_cast : memref<16x16xf16>
      %copy_out = hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws_t : tensor<16x16xf16>) -> tensor<16x16xf16>

      %dest2 = tensor.empty() : tensor<16x16xf16>
      %dot2 = hivm.hir.mmadL1 ins(%copy_out, %tensorK, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<1x1x16x16xf16>, i1, index, index, index) outs(%dest2 : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.fixpipe ins(%dot2 : tensor<16x16xf16>) outs(%gm_dst : memref<16x16xf16>)
    }
    return
  }
}
