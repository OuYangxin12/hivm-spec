// L0 注入例（M5/T5.6）：矩阵乘的 B 输入被本地新建的零张量顶替。
// 真实来源：dcv 前端 pass 在 V/C 拆分时丢失 tl.dot 的 B 跨核通路，用零张量
// 兜底 —— 点积恒 0（语义错，但 IR 良构、可编译、不报错，故结构验证必然漏过）。
// B 与 outs 同时指向同一个新零张量，是这一形态的签名。
module {
  func.func @neutralized_matmul_input(%gm: memref<16x16xf32, #hivm.address_space<gm>>) {
    %cst = arith.constant 0.000000e+00 : f32
    %a = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%gm : memref<16x16xf32, #hivm.address_space<gm>>) outs(%a : memref<16x16xf32, #hivm.address_space<ub>>)
    %ta = bufferization.to_tensor %a : memref<16x16xf32, #hivm.address_space<ub>>
    %empty = tensor.empty() : tensor<16x16xf32>
    %zero = linalg.fill ins(%cst : f32) outs(%empty : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = linalg.matmul ins(%ta, %zero : tensor<16x16xf32>, tensor<16x16xf32>) outs(%zero : tensor<16x16xf32>) -> tensor<16x16xf32>
    return
  }
}
