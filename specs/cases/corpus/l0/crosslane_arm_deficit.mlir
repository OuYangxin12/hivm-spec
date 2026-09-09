// L0 注入微例：**跨泳道** arm 供给短缺 —— wait 在 PIPE_V（消费道）、
// set 在 PIPE_MTE2（生产道），每迭代 2 wait : 1 set（16:8）。
//
// 为何单列此例（M2 审查发现 1）：这是流水线握手的**常态形态**（生产者 set /
// 消费者 wait）。异道 set 永远可达 ⇒ 规则 B 的"set 不可达"前提不成立，
// 规则 A 的"无人 set"也不成立——本例曾被判 OK/exit 0（假阴性），而 7 个 wait
// 实际永久饿死。规则 C（arm-deficit）以纯计数论证补上该盲区：
// 16 wait > 8 set + 1 初始装载 ⇒ 必有 7 个 wait 饿死，与交错顺序无关。
//
// 判定预期：DEADLOCK / 规则 C（arm-deficit），定位到第 9 次 wait（首个必然饿死）。
// 对照面见 crosslane_balanced_handshake.mlir（同形态但配平 ⇒ 必须 OK）。
module {
  func.func @crosslane_arm_deficit() {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c128 step %c16 {
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
    }
    return
  }
}
