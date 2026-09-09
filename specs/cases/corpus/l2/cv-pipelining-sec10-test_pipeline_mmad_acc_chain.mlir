module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline_mmad_acc_chain(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %inA: memref<16x16xf16>, %inB1: memref<16x16xf16>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    // harvested-away: inA 提升为函数参数
    %A = bufferization.to_tensor %inA : memref<16x16xf16>
    // harvested-away: inB1 提升为函数参数
    %B1 = bufferization.to_tensor %inB1 : memref<16x16xf16>
    %unrelated_in = tensor.empty() : tensor<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = arith.constant 4 : i32
    scf.for %i = %c0 to %bound step %step : i32 {
      %init_complex = tensor.empty() : tensor<16x16xf16>
      %dot0 = hivm.hir.mmadL1 ins(%A, %B1, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%init_complex : tensor<16x16xf16>) -> tensor<16x16xf16>

      %ub = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%unrelated_in : tensor<16x16xf16>) outs(%ub : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub_cast = memref.memory_space_cast %ub : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %ub_t = bufferization.to_tensor %ub_cast : memref<16x16xf16>
      %vd = tensor.empty() : tensor<16x16xf16>
      %vexp = hivm.hir.vexp ins(%ub_t : tensor<16x16xf16>) outs(%vd : tensor<16x16xf16>) -> tensor<16x16xf16>
      %cb = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %cb_cast = memref.memory_space_cast %cb : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%vexp : tensor<16x16xf16>) outs(%cb_cast : memref<16x16xf16>)
      %B2 = bufferization.to_tensor %cb_cast : memref<16x16xf16>

      %dot1 = hivm.hir.mmadL1 ins(%A, %B2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot0 : tensor<16x16xf16>) -> tensor<16x16xf16>
      %out_buf = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot1 : tensor<16x16xf16>) outs(%out_buf : memref<16x16xf16, #hivm.address_space<ub>>)
    }
    return
  }
}
