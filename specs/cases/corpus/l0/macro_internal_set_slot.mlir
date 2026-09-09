// L0 微例（T3.0 对拍后补）：不透明 macro 的内部 set —— sync_event_slot<..., set, <EVENT_ID0>>
// 表示 set 发生在 macro **内部**（IR 不可见），GSS 在 macro 后注入 wait_flag。
// 健康对照：IR 里只见一个孤立 wait，但 macro 提供了 set，故不应判死锁。
// 该形态若不解析槽位，判定只能靠 INITIAL_ARM=1 侥幸放行（理由错、结论对）；
// 一旦初态改 0 立刻变假阳性。详见 docs/crosscheck/T3.0-flag-semantics.md §5。
module {
  func.func @macro_internal_set_slot(%gm: memref<16xf16, #hivm.address_space<gm>>) {
    %ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%gm : memref<16xf16, #hivm.address_space<gm>>) outs(%ub : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.custom_macro
        {hivm.tcore_type = #hivm.tcore_type<VECTOR>,
         hivm.pipe_in = #hivm.pipe<PIPE_MTE2>,
         hivm.pipe_out = #hivm.pipe<PIPE_V>,
         symbol = "k_custom_macro_set",
         sync_event_slots = [
           #hivm.sync_event_slot<#hivm.pipe<PIPE_MTE2>, #hivm.pipe<PIPE_MTE1>, set, <EVENT_ID0>>
         ]}
        "user.macro_set"
        ins(%ub : memref<16xf16, #hivm.address_space<ub>>)
        outs(%ub : memref<16xf16, #hivm.address_space<ub>>)
    hivm.hir.wait_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
    return
  }
}
