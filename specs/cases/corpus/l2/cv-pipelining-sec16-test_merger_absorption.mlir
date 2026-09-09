module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_merger_absorption(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %input1: memref<16x16xf16>, %input2: memref<?xf16>, %gm_dst: memref<16x16xf16>) attributes {WorkspaceArgIdx = 0 : i16, func_dyn_memref_args = dense<[true]> : vector<1xi1>, global_kernel = "local", hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix"} {
    // harvested-away: input1 提升为函数参数
    %tensor1 = bufferization.to_tensor %input1 : memref<16x16xf16>
    // harvested-away: input2 提升为函数参数
    %initin = memref.reinterpret_cast %input2 to offset: [0], sizes: [16, 16], strides: [16, 1] : memref<?xf16> to memref<16x16xf16>
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %c16 = arith.constant 16 : index
    %step = arith.constant 2 : i32
    %bound = arith.constant 4 : i32
    %vinit = tensor.empty() : tensor<16x16xf16>
    %cond = arith.constant false
    // harvested-away: gm_dst 提升为函数参数
    scf.for %i = %c0 to %bound step %step iter_args(%acc = %vinit) -> (tensor<16x16xf16>) : i32 {
      %alloc = memref.alloc() : memref<16x16xf16>
      hivm.hir.load ins(%initin : memref<16x16xf16>) outs(%alloc : memref<16x16xf16>)
      %tensor2 = bufferization.to_tensor %alloc : memref<16x16xf16>
      %dest = tensor.empty() : tensor<16x16xf16>
      %dot = hivm.hir.mmadL1 ins(%tensor1, %tensor2, %true, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%dest : tensor<16x16xf16>) -> tensor<16x16xf16>
      %ub0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
      hivm.hir.fixpipe ins(%dot : tensor<16x16xf16>) outs(%ub0 : memref<16x16xf16, #hivm.address_space<ub>>)
      %ub0_cast = memref.memory_space_cast %ub0 : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %wst = bufferization.to_tensor %ub0_cast : memref<16x16xf16>

      %vdest = tensor.empty() : tensor<16x16xf16>
      %if = scf.if %cond -> tensor<16x16xf16> {
        %new = hivm.hir.vexp ins(%wst : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>
        scf.yield %new : tensor<16x16xf16>
      } else {
        scf.yield %wst : tensor<16x16xf16>
      }

      %ws1 = memref.alloc() : memref<16x16xf16, #hivm.address_space<cbuf>>
      %ws1_cast = memref.memory_space_cast %ws1 : memref<16x16xf16, #hivm.address_space<cbuf>> to memref<16x16xf16>
      hivm.hir.copy ins(%if : tensor<16x16xf16>) outs(%ws1_cast : memref<16x16xf16>)

      %is_first = arith.cmpi eq, %i, %c0 : i32
      %sel = arith.select %is_first, %vinit, %if : tensor<16x16xf16>
      scf.yield %sel : tensor<16x16xf16>
    }
    return
  }
}
