// L0 微例：跨迭代事件对 —— set_flag/wait_flag 跨循环迭代配对（D11 spike 的 op ②）。
// 承载结构：VSync(event_id + set/wait pipe) + 顺序敏感的配对判定。
module {
  func.func @cross_iter_event_pair(%gm: memref<128xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %ub = memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
    scf.for %i = %c0 to %c128 step %c16 {
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.load ins(%gm : memref<128xf32, #hivm.address_space<gm>>) outs(%ub : memref<128xf32, #hivm.address_space<ub>>)
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    }
    return
  }
}
