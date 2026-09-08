// L0 微例：嵌套区域 —— 验证"先本区域 nodes 再进子区域"的确定性遍历约定。
// 承载结构：VRegion 嵌套 + VLoop 嵌套 + node_order() 稳定性。
module {
  func.func @nested_loop_scope(%gm: memref<64xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %ub = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    %ub2 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    scf.for %i = %c0 to %c2 step %c1 {
      hivm.hir.load ins(%gm : memref<64xf32, #hivm.address_space<gm>>) outs(%ub : memref<64xf32, #hivm.address_space<ub>>)
      scf.for %j = %c0 to %c2 step %c1 {
        hivm.hir.vmul ins(%ub, %ub : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) outs(%ub2 : memref<64xf32, #hivm.address_space<ub>>)
      }
    }
    return
  }
}
