module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @test_pipeline_nested_static_counter() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c16 = arith.constant 16 : index
  %true = arith.constant true
  %input = tensor.empty() : tensor<16x16xf16>
  %out = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>

  %init_static = tensor.empty() : tensor<16xf16>

  scf.for %outer = %c0 to %c4 step %c1 : index {
    %counter = memref.alloca() {normalize_matmul_counter} : memref<i32>
    memref.store %c0_i32, %counter[] : memref<i32>

    %cube = scf.for %inner = %c0 to %c4 step %c1 iter_args(%iter = %init_static) -> tensor<16xf16> {
      %count = memref.load %counter[] : memref<i32>
      %next = arith.addi %count, %c1_i32 : i32
      memref.store %next, %counter[] : memref<i32>

      %first = arith.cmpi eq, %count, %c0_i32 : i32
      %dot = hivm.hir.mmadL1 ins(%input, %input, %first, %c16, %c16, %c16 : tensor<16x16xf16>, tensor<16x16xf16>, i1, index, index, index) outs(%input : tensor<16x16xf16>) -> tensor<16x16xf16>

      scf.yield %iter : tensor<16xf16>
    }

    hivm.hir.fixpipe ins(%input : tensor<16x16xf16>) outs(%out : memref<16x16xf16, #hivm.address_space<ub>>)
    %out_cast = memref.memory_space_cast %out : memref<16x16xf16, #hivm.address_space<ub>> to memref<16x16xf16>
    %tensor = bufferization.to_tensor %out_cast : memref<16x16xf16>

    %vdest = tensor.empty() : tensor<16x16xi32>
    %ext_read = memref.load %counter[] : memref<i32>
    %vector = hivm.hir.vbrc ins(%ext_read : i32) outs(%vdest : tensor<16x16xi32>) -> tensor<16x16xi32>
  }
  return
}
}
