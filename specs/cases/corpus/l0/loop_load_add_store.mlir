// L0 微例：最小 UB 占用回路 —— 静态 shape、单层 scf.for、load→vadd→store。
// 承载结构：VLoop(静态 trip_count) / VAlloc(STATIC_SHAPE) / 读写效应。
module {
  func.func @loop_load_add_store(%gm: memref<256xf32, #hivm.address_space<gm>>) {
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c64 = arith.constant 64 : index
    %ub_a = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    %ub_b = memref.alloc() : memref<256xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%gm : memref<256xf32, #hivm.address_space<gm>>) outs(%ub_a : memref<256xf32, #hivm.address_space<ub>>)
    scf.for %i = %c0 to %c256 step %c64 {
      hivm.hir.vadd ins(%ub_a, %ub_a : memref<256xf32, #hivm.address_space<ub>>, memref<256xf32, #hivm.address_space<ub>>) outs(%ub_b : memref<256xf32, #hivm.address_space<ub>>)
    }
    hivm.hir.store ins(%ub_b : memref<256xf32, #hivm.address_space<ub>>) outs(%gm : memref<256xf32, #hivm.address_space<gm>>)
    return
  }
}
