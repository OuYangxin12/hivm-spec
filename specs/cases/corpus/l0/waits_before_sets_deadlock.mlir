// L0 微例：wait 全部先于首个 set —— 真实故障的最小复现
// （CreatePreload stage-major 死锁，A5 MixedCV 全核 timeout）。
// 关键：set/wait 计数完全配平（3:3），FileCheck 类结构验证必然漏过；
// 只有顺序视图能暴露"3 个 wait 先于首个 re-arm set"。
// 承载结构：VModule.sync_order() 的顺序判定能力。
module {
  func.func @waits_before_sets_deadlock() {
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 15
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 14
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 14
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE2>, <PIPE_S>] flag = 14
    return
  }
}
