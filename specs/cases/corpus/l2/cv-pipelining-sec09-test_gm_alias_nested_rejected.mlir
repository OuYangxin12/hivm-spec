module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_gm_alias_nested_rejected(%gmArg: memref<16x16xf16>, %input1: memref<16x16xf16>, %input2: memref<16x16xf16>) attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    // harvested-away: input1 提升为函数参数
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    // harvested-away: input2 提升为函数参数
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = arith.constant 4 : i32
    %cond = arith.constant false
    scf.for %i = %c0 to %bound step %step : i32 {
      %allocC = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%input2 : memref<16x16xf16>) outs(%allocC : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %allocC : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      scf.if %cond {
        hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%gmArg : memref<16x16xf16>)
      }

      %allocV = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      %allocV_cast = memref.memory_space_cast %allocV : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      hivm.hir.load ins(%gmArg : memref<16x16xf16>) outs(%allocV_cast : memref<16x16xf16>)
      %tv = bufferization.to_tensor %allocV_cast : memref<16x16xf16>
      %vdest = tensor.empty() : tensor<16x16xf16>
      %exp = hivm.hir.vexp ins(%tv : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%exp : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)
    }
    return
  }
}
