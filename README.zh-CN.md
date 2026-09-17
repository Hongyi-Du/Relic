# Relic（中文说明）

[English README](README.md) · [发布范围](docs/release-scope.md) · [论文结果](artifacts/paper_results/paper_results.md) · [已知缺口](docs/KNOWN_RELEASE_GAPS.md)

`relic` 是论文的研究代码与复现实验仓库，不是通用 SDK。它包含 OrgEnv、B0--B3、十个 `relic-main-v1` workload、主实验与 transfer 入口、evaluator 边界、论文 aggregate results 以及公开 Inspector。完整的历史论文运行目前不能在此 checkout 中复现：正式模型运行仍等待作者发布 evaluator bindings 和证据资产；请先阅读[已知缺口](docs/KNOWN_RELEASE_GAPS.md)及[实际运行报告](docs/REPRODUCTION_RUN_REPORT.md)。

## 分支与仓库导航

| 分支 | 用途 |
| --- | --- |
| `main` | 干净、canonical 的论文复现分支 |
| `hci` | `main` 加 P2/P3 human-seat HCI 扩展；切换后读 `docs/HCI_GUIDE.md` |
| `cooper` | `main` 加 CooperBench B3-2 扩展；切换后读 `reproduction/cooperbench/README.md` |
| `full-tests` | 未来的清理后核心历史回归测试分支；尚未发布，不能切换 |

通用、与论文 benchmark 无关的组织 runtime 位于 [Relic-Agent](https://github.com/Hongyi-Du/Relic-Agent)。作者尚未提供项目网站 URL，因此本仓库不会猜测或伪造网页链接。

## B0--B3 条件

| Arm | 成员 | 决策 | profile/capability conditioning | institutionalization |
| --- | ---: | --- | --- | --- |
| B0 | 1 | direct LLM | 关闭 | 关闭 |
| B1 | 8 个持久成员 | direct LLM | 关闭 | 关闭 |
| B2 | 8 个持久成员 | SDL/profile policy | 开启 | 关闭 |
| B3 | 8 个持久成员 | SDL/profile policy | 开启 | 开启（包括 runtime protocol binding） |

YAML config 是权威定义：`configs/arms/`。主 benchmark 在 [`benchmarks/relic-main-v1/`](benchmarks/relic-main-v1/)，最终论文 aggregate summary 在 [`artifacts/paper_results/`](artifacts/paper_results/)。

## Linux / WSL2 快速开始

正式 runtime 是 Linux；Windows 用户请在 WSL2 的 Linux filesystem（例如 `~/relic`）中 clone，不要在 `/mnt/c` 下运行。PowerShell 仅调用 WSL2 launcher，不直接运行 Relic core。

```bash
git clone <YOUR-RELIC-REMOTE> relic
cd relic
cp .env.example .env
uv sync --extra dev --frozen
uv run relic verify-benchmark
uv run relic check-env --scope core
uv run relic smoke --mode mock
```

以上命令不调用模型。输出默认写入 `outputs/`；可通过 `RELIC_OUTPUT_ROOT` 或显式 `--output-root` 设置。正式 evaluator/image binding 尚未发布，因此 `check-env --scope formal`、正式 smoke 和非 dry-run 论文运行会按设计 fail-closed。

生成一个不调用 provider 的 120-cell GPT dry plan：

```bash
uv run relic run-main \
  --model gpt-5.6-terra \
  --output-root outputs/main-study \
  --manifest outputs/main-study/source_main_manifest.json \
  --max-parallel 1 \
  --dry-run
```

完整 `run-main` 是高成本、高内存命令，不能作为安装验证。参数优先级固定为：CLI 显式参数 → canonical config → 文档化环境变量 → repository default；B0--B3、workload、seed、tick 和 evaluator policy 不会被 `.env` 偷偷改写。

## Docker 快速开始

Docker 与本地路径调用同一个 `relic` CLI。它不写入 API key、历史结果、cache 或 selected trace。

```bash
cp .env.example .env
mkdir -p outputs cache traces
docker compose build relic
docker compose run --rm relic check-env --scope core
docker compose run --rm relic smoke --mode mock
```

详见[环境与平台支持](docs/environment.md)：其中包含所有环境变量、Docker volume、WSL2/PowerShell 使用方式、并发与内存策略。

## Inspector 与论文结果

```bash
uv run relic replay --trace /path/to/selected-trace.json
uv run relic inspect --trace /path/to/selected-trace.json
```

Inspector 默认只绑定 `127.0.0.1:8765`。它只读取经过严格校验的 `relic-trace-v1`，不读取 checkpoint、private memory、provider messages 或 evaluator workspace。作者尚未提供已脱敏的 selected paper trace；不要用 aggregate result 或 count-only sidecar 伪造它。

可复制的正式入口、输出结构、transfer、CooperBench 边界以及当前不可补齐的外部资产，请从 [English README](README.md) 和 `docs/` 中的链接继续阅读。
