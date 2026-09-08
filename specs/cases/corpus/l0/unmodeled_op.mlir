// L0 微例：未建模 op —— COVERAGE_GAP 的诚实降级路径（不变量 4 / FR4）。
// 此处用真实存在但描述库暂不建模的 op（vexp），验证工具报缺口而非静默通过。
// 注意：不使用未注册假 op（那需 allow-unregistered，属 lit 夹具，见 D13）。
module {
  func.func @unmodeled_op(%gm: memref<32xf32, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    %ub2 = memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%gm : memref<32xf32, #hivm.address_space<gm>>) outs(%ub : memref<32xf32, #hivm.address_space<ub>>)
    hivm.hir.vexp ins(%ub : memref<32xf32, #hivm.address_space<ub>>) outs(%ub2 : memref<32xf32, #hivm.address_space<ub>>)
    return
  }
}
