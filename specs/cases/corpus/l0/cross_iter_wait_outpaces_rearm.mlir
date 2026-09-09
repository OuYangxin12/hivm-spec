// L0 注入微例：跨迭代 re-arm 不足 —— 循环体内 2 次 wait 对 1 次 set（同事件）。
// set/wait 总数不配平（16:8），FileCheck 类结构验证若逐 op 数 sync 也可能漏过
// "哪个多哪个少"的语义；时序引擎的 re-arm 模型（初始装载 +1）+ 泳道程序序
// 判它为 wait-for 环（结构性死锁，T2.3 规则 B）。
// 注入缺陷对应 AC1 死锁维（M2）：跨迭代配对错位（wait 消费速度 > re-arm 速度）。
// 承载结构：VSync(event_id) + 泳道内顺序敏感 + scf.for 静态 trip 全量展开。
module {
  func.func @cross_iter_wait_outpaces_rearm() {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c128 step %c16 {
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE2>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    }
    return
  }
}
