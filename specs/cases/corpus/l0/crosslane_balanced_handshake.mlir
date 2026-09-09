// L0 健康对照：跨泳道**配平**握手 —— wait 在 PIPE_V、set 在 PIPE_MTE2，
// 每迭代 1 wait : 1 set（8:8）。这是真实流水线双缓冲握手的最小健康形态。
//
// 为何必须入库（M2 审查发现 1 的配套防线）：规则 C（arm-deficit）是**放宽
// 判定**方向的变更（新增 DEADLOCK 触发路径），最大风险是把健康的跨泳道流水
// 误判为死锁（假阳性，NFR2 优先压制的正是这一侧）。本例与
// crosslane_arm_deficit.mlir 仅差一次 set，构成规则 C 的判定边界对照：
//   16w:8s → DEADLOCK（短缺 7）    8w:8s+1 初装 → OK（供给充足）
//
// 判定预期：OK，且 arm_supply 为空（无短缺事件）。
module {
  func.func @crosslane_balanced_handshake() {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c128 step %c16 {
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    }
    return
  }
}
