module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_skip_unobserved_counter_clone() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : index
    %true = arith.constant true
    %input = tensor.empty() : tensor<16x16xf16>
    %init = tensor.empty() : tensor<16x16xf16>
    %out = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>

    scf.for %outer = %c0 to %c4 step %c1 : i32 {
      %counter = memref.alloca() {normalize_matmul_counter} : memref<i32>
      memref.store %c0, %counter[] : memref<i32>

      %cube = scf.if %true -> tensor<16x16xf16> {
        %count = memref.load %counter[] : memref<i32>
        %first = arith.cmpi eq, %count, %c0 : i32
        %dot = hivm.hir.mmadL1 ins(%input, %input, %first, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%init : tensor<16x16xf16>) -> tensor<16x16xf16>
        %next = arith.addi %count, %c1 : i32
        memref.store %next, %counter[] : memref<i32>
        scf.yield %dot : tensor<16x16xf16>
      } else {
        scf.yield %init : tensor<16x16xf16>
      }

      hivm.hir.fixpipe ins(%cube : tensor<16x16xf16>) outs(%out : memref<16x16xf16, #hivm.address_space<ub>>)
      %out_cast = memref.memory_space_cast %out : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
      %tensor = bufferization.to_tensor %out_cast : memref<16x16xf16>
      %vdest = tensor.empty() : tensor<16x16xf16>
      %vector = hivm.hir.vexp ins(%tensor : tensor<16x16xf16>) outs(%vdest : tensor<16x16xf16>) -> tensor<16x16xf16>

    }
    return
  }
}
