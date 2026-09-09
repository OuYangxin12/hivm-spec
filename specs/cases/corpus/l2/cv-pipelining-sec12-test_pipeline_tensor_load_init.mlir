module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_pipeline_tensor_load_init(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %gm_dst: memref<16x16xf16>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true, true]> : vector<2xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    %a = tensor.empty() : tensor<16x16xf16>
    %vector_src = tensor.empty() : tensor<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = arith.constant 4 : i32
    scf.for %i = %c0 to %bound step %step : i32 {
      %vector_init = tensor.empty() : tensor<16x16xf16>
      %vector_value = hivm.hir.vexp ins(%vector_src : tensor<16x16xf16>) outs(%vector_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      %workspace = memref_ext.alloc_workspace() from %arg0 : from memref<?xi8> to memref<16x16xf16>
      annotation.mark %workspace {hivm.multi_buffer = 2 : i32} : memref<16x16xf16>
      %workspace_tensor = bufferization.to_tensor %workspace restrict writable : memref<16x16xf16>
      %stored = hivm.hir.store ins(%vector_value : tensor<16x16xf16>) outs(%workspace_tensor : tensor<16x16xf16>) {"inserted-store"} -> tensor<16x16xf16>

      %load_init = tensor.empty() : tensor<16x16xf16>
      %loaded = hivm.hir.load ins(%stored : tensor<16x16xf16>) outs(%load_init : tensor<16x16xf16>) {"inserted-load"} init_out_buffer = false core_type = <CUBE> -> tensor<16x16xf16>
      %dot_init = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%a, %loaded, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dot_init : tensor<16x16xf16>) -> tensor<16x16xf16>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%gm_dst : memref<16x16xf16>)
    }
    return
  }
}
