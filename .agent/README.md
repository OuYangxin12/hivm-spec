# `.agent/` — 本仓的 agent skill

只有一个 skill：[`hivm-spec/`](hivm-spec/SKILL.md)。**面向使用者**（consumer）——教 agent
怎么用 hivm-spec 验证 HIVM pass，不复制仓库开发规约。

## 与其它文档的分工

| 你要做的事 | 去哪 |
|---|---|
| 用工具验证一个 pass（跑什么、结论怎么读、边界在哪） | `.agent/skills/hivm-spec/SKILL.md` |
| 改本仓 `src/` / `specs/`（门禁、治理、架构不变量） | 仓库根 [`AGENTS.md`](../../AGENTS.md)（唯一权威） |
| 需求与决策依据 | `docs/requirements.md`、`docs/design-framework.md` |
| 按里程碑推进 | `docs/milestone-plan.md`、`docs/tasks/` |

skill 里凡与 `AGENTS.md` 冲突之处，以 `AGENTS.md` 为准。skill 是操作视图，不是纪律来源。

## 维护这个 skill

1. **不要写死实测数字。** 覆盖率/可判率是语料快照上的值。要引用就跑
   `scripts/verify-skill.sh`——它校验 skill 里全部断言（含等价可判率现测）。
2. 改了 `specs/` 描述或信任级别 → 跑 `verify-skill.sh`；有断言失效就同步改 skill，
   **不要**把失效的断言留在文档里。
3. **改 `src/` 后按 AGENTS.md §1 跑门禁**（ruff / mypy / pytest / spec-gate）。skill 自带的
   `verify-skill.sh` 只校验**文档断言**，不替代仓内门禁。
4. 更新 `SKILL.md` front-matter 的 `metadata.codebase_commit` 与 `generated`，否则下个
   agent 无法判断这份 skill 有多旧。
5. skill 的每条断言都要能指回一处代码或一次实测。指不回的就删。
