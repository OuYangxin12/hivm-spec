# 环境与安装（四个必错点）

先跑 `python -m hivm_spec doctor`。它把下面四件事全部探测一遍，缺什么给什么 remedy。

## 分层：核心层 vs IR 接口层

| 层 | 依赖 | 能跑什么 |
|---|---|---|
| 核心层 | 纯 Python ≥ 3.10 | `doctor`、`check`、`gen`、核心单测（`pytest -m "not requires_bindings"`） |
| IR 接口层 | **Python 3.10 + bishengir bindings（cp310 ABI）** | `run`、`tool *`、`requires_bindings` 测试 |

`requires_bindings` 的测试在无绑定环境**跳过并登记为覆盖缺口**——不是失败，也不是通过。

## 四个必错点

1. **Python 版本必须是 3.10。** bindings 是 cp310 ABI，换 3.11/3.12/3.13 直接 import 失败。
   核心层用系统解释器没问题，`run`/`tool` 必须 3.10。
2. **`HIVM_SPEC_BINDINGS` 必须指向绑定树。** 它指定 bishengir bindings 的位置：

   ```bash
   export HIVM_SPEC_BINDINGS="$PWD/.bindings"
   ```

   不设 → `doctor` 报 bindings missing → `run`/`tool` 全部不可用。
3. **解释器安装目录要持久。** uv 的解释器若落在 `/tmp` 会随重启丢失，使 `.venv310` 断链：

   ```bash
   export UV_PYTHON_INSTALL_DIR="$PWD/.uvpython"
   ```
4. **两个可选 extra 各管一档，缺失时不得互相顶替。**

   - 缺 `ml_dtypes` → bf16/fp8 不可用。**禁止**用 f32 顶替：f32 算 bf16 会给出比真实硬件
     *更精确* 的结果，对拍假通过。
   - 缺 `z3` → 符号档不可用。**禁止**"退回具体档"：两档结论强度不同，不可互替。

## 装到一台新机器（skill 自带脚本）

```bash
bash scripts/install.sh --repo <hivm-spec 本地仓>     # 或 --git git+https://…
```

四步：找 3.10 解释器 → 装核心层 → 生成配置文档 → 探测 bindings，末尾打印**可直接用的
一句话入口**。它把"核心层装上了"与"六类功能可用"**分开报告**——以后者为准。

单独定位 bindings（真 import 校验，不改环境）：

```bash
python .agent/skills/hivm-spec/scripts/bootstrap.py            # 探测 + 报告
eval "$(python …/bootstrap.py --shell)"                        # 成功时直接 export
```

候选来源按可信度：`$HIVM_SPEC_BINDINGS` → hivm-spec 仓 `.bindings` → 祖先目录里的
主仓构建树 → `$ASCEND_*` 指向的 toolkit。**不做全盘猜路径**：猜出来的候选会带来
"看着像但其实不是"的树。

配置文档定位优先级：`-c` > CWD 的 `build/config.json` > `$HIVM_SPEC_CONFIG`。
离开仓目录调用时设 `HIVM_SPEC_CONFIG`（否则缺省的 CWD 相对路径找不到）。



```bash
export UV_CACHE_DIR=/tmp/uvcache                      # 只读 HOME 时必设
export UV_PYTHON_INSTALL_DIR="$PWD/.uvpython"

uv pip install -e ".[test,dev]"                       # 核心层
uv venv .venv310 --python 3.10
uv pip install -e ".[test]" --python .venv310/bin/python
bash scripts/setup_bindings.sh                        # 绑定树落位 .bindings/（约 246M）

# 可选能力
uv pip install -e '.[symbolic]' --python .venv310/bin/python   # z3：符号档
uv pip install -e '.[numeric]'  --python .venv310/bin/python   # ml_dtypes：bf16/fp8
```

## 主仓侧（只想用工具，不改本仓）

```bash
cd <hivm-spec 仓>
export HIVM_SPEC_BINDINGS="$PWD/.bindings"
alias hivm-spec='python -m hivm_spec'    # 或 pip install -e . 后用 hivm-spec
```

## doctor 输出解读

```
hivm-spec 环境自检
  ✓ Python                 3.10.21（cp310，IR 接口层可用）
  ✓ bishengir bindings     可用（HIVM_SPEC_BINDINGS=/…/.bindings）
  ✓ numpy                  2.2.6
  ✓ ml_dtypes              0.5.4
  ✓ z3                     5.1.0.0

结论：全部可用
```

状态三档：`ok` / `degraded`（缺了只影响部分能力，工具仍可用）/ `missing`（核心能力不可用）。
退出码：**有 MISSING → 1；仅 DEGRADED → 0。** 所以 exit 0 不等于全能力可用，要看逐项。

doctor 的纪律与工具同源：缺 `ml_dtypes` 不会说"可用 f32 替代"，缺 z3 不会说"可退回具体
档"。**缺口是合法结论，静默不是。**
