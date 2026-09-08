// L0 微例（注入缺陷）：两个 64K 元素 f32 buffer 各 256KB，生存期完全重叠，
// UB 容量 192KB —— 峰值 512KB，必然溢出。
// 承载 AC1 的"溢出维"：这是唯一一个刻意违反容量约束的语料。
// 注意它与"负例语料"（expected-error）不同：本文件语法完全合法、可严格解析，
// 违反的是**语义约束**，正是 spec 工具该抓而 FileCheck 抓不到的东西。
module {
  func.func @ub_overflow_injected(%gm: memref<65536xf32, #hivm.address_space<gm>>) {
    %ub_a = memref.alloc() : memref<65536xf32, #hivm.address_space<ub>>
    %ub_b = memref.alloc() : memref<65536xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%gm : memref<65536xf32, #hivm.address_space<gm>>) outs(%ub_a : memref<65536xf32, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%ub_a, %ub_a : memref<65536xf32, #hivm.address_space<ub>>, memref<65536xf32, #hivm.address_space<ub>>) outs(%ub_b : memref<65536xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%ub_b : memref<65536xf32, #hivm.address_space<ub>>) outs(%gm : memref<65536xf32, #hivm.address_space<gm>>)
    return
  }
}
