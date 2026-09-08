// L0 微例：动态 shape —— 尺寸未知须成为一等公民（实测 123/189 语料含动态 shape）。
// 承载结构：VAlloc(size_origin=UNKNOWN) + UNKNOWN_SIZE 缺口路径。
module {
  func.func @dynamic_shape_alloc(%gm: memref<?xf32, #hivm.address_space<gm>>, %n: index) {
    %ub = memref.alloc(%n) : memref<?xf32, #hivm.address_space<ub>>
    hivm.hir.load ins(%gm : memref<?xf32, #hivm.address_space<gm>>) outs(%ub : memref<?xf32, #hivm.address_space<ub>>)
    hivm.hir.store ins(%ub : memref<?xf32, #hivm.address_space<ub>>) outs(%gm : memref<?xf32, #hivm.address_space<gm>>)
    return
  }
}
