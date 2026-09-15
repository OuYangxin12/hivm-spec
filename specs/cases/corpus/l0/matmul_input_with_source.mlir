// L0 健康对照（M5/T5.6）：同一形状的矩阵乘，但两个输入都有真实来源
// （各自从 GM 载入 UB），零只出现在 outs（累加器）槽位。
// 对照目的：证明 operand_wiring 不会因"看到零填充"就误报（FR2）。
module {
  func.func @matmul_input_with_source(%gm: memref<16x16xf32, #hivm.address_space<gm>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %a = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    %b = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%gm : memref<16x16xf32, #hivm.address_space<gm>>) outs(%a : memref<16x16xf32, #hivm.address_space<ub>>)
    hivm.hir.load ins(%gm : memref<16x16xf32, #hivm.address_space<gm>>) outs(%b : memref<16x16xf32, #hivm.address_space<ub>>)
    %ta = bufferization.to_tensor %a : memref<16x16xf32, #hivm.address_space<ub>>
    %tb = bufferization.to_tensor %b : memref<16x16xf32, #hivm.address_space<ub>>
    %empty = tensor.empty() : tensor<16x16xf32>
    %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = linalg.matmul ins(%ta, %tb : tensor<16x16xf32>, tensor<16x16xf32>) outs(%zero : tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}
