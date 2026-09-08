# specs/ — 语义域描述库

Agent（与人类）编写的 HIVM 语义域描述：op 值语义与效应（`ops` 段）、
VM 状态模型（`vm` 段：地址空间/pipe/event/容量）、要生成的工具声明（`checks` 段）。

规则：
- 描述为 Python 模块，进 PR 评审；`trust` 升级必须附对拍证据（canonical / 与主仓 C++ 链对拍 / Hypothesis 性质）。
- 引用不完备或签名不一致将被生成器静态检查拒绝。
- 架构差异（A3/A5）走 `arch` 参数化，禁止两份全量描述。
