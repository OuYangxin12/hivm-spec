// L0 注入微例：**set/wait 总数完全配平**（16:16），但同泳道内 2 个 wait
// 连排先于 2 个 set —— R2 台账记录的 CreatePreload stage-major 乱序死锁形态。
//
// 立论价值（design-framework §13 R2 / M2 卡设计要点 4）：计数配平 ⇒ FileCheck
// 类结构验证必然漏过（"哪个多哪个少"看不出问题），而顺序不可行只有时序模型
// 抓得住。泳道程序序门控使两个 set 被前置受阻 wait 永久挡住 → wait-for 环。
//
// 判定预期：DEADLOCK / 规则 B（wait-cycle）。与 cross_iter_wait_outpaces_rearm
// （16:8 不配平，规则 B）互补——本例证明判定不依赖计数失衡。
// 承载结构：VSync(event_id) + 同泳道顺序敏感 + scf.for 静态 trip 全量展开。
module {
  func.func @balanced_misordered_same_lane() {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c128 step %c16 {
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    }
    return
  }
}
