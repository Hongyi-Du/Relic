# Relic（中文说明）

[English README](README.md) · [发布范围](docs/release-scope.md) · [论文结果](artifacts/paper_results/paper_results.md) · [已知缺口](docs/KNOWN_RELEASE_GAPS.md)

`relic` 是论文的研究代码与复现实验仓库，包含 OrgEnv、B0--B3、十个 `relic-main-v1` workload、主实验与 transfer 入口、evaluator 和公开 Inspector。填写自己的 API key、gateway 和模型名后即可运行，默认使用仓库公开的本地 evaluator，无需 evaluator binding。

## 分支与仓库导航

| 分支 | 用途 |
| --- | --- |
| `main` | 干净、canonical 的论文复现分支 |
| `hci` | `main` 加 P2/P3 human-seat HCI 扩展；切换后读 `docs/HCI_GUIDE.md` |
| `cooper` | `main` 加 CooperBench B3-2 扩展；切换后读 `reproduction/cooperbench/README.md` |

通用、可配置的组织 runtime 位于 [Relic-Agent](https://github.com/Hongyi-Du/Relic-Agent)。普通测试直接保留在仓库中，不需要额外的测试分支。

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
git clone https://github.com/Hongyi-Du/Relic.git relic
cd relic
cp .env.example .env
uv sync --extra dev --frozen
uv run relic verify-benchmark
uv run relic check-env --scope core
uv run relic smoke --mode mock
```

以上命令不调用模型。输出默认写入 `outputs/`；可通过 `RELIC_OUTPUT_ROOT` 或显式 `--output-root` 设置。在 `.env` 填写自己的路由：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-gateway.example/v1
RELIC_RUNTIME_MODEL=your-provider-deployment
```

直接运行 `uv run relic ...` 时，命令只继承当前进程环境，不会隐式加载
`.env`。需要调用 provider 的命令请显式使用
`uv run --env-file .env relic ...`（下方示例如此）；`scripts/bash/` 下的 Bash
wrapper 会自动加载仓库根目录 `.env` 中允许的变量。

CLI 的 `--runtime-model` 可以覆盖 runtime 模型名，`--model` 保留论文模型标识。Claude 分组也可以设置 `RELIC_CLAUDE_OPUS_4_6_MODEL`，不需要恢复作者的私人 alias。

生成一个不调用 provider 的 120-cell GPT dry plan：

```bash
uv run relic run-main \
  --model gpt-5.6-terra \
  --output-root outputs/main-study \
  --manifest outputs/main-study/source_main_manifest.json \
  --max-parallel 1 \
  --dry-run
```

配置好模型后，移除 `--dry-run` 执行主实验：

```bash
uv run --env-file .env relic run-main --model gpt-5.6-terra --output-root outputs/main-study --max-parallel 1
```

主实验仍保留每模型 120 cells、每 cell 336 ticks 的论文设置。各 batch 的 `experiment_runs.json/jsonl` 汇总 B0–B3 的实验记录、评分证据与执行状态，顶层 `source_main_manifest.json` 链接这些结果。只有主动添加 `--strict-reproducibility` 时才要求完整 binding、digest、platform 和 hash；普通运行会记录本地实际观察到的 evaluator metadata。完整输出与 legacy cell aggregate 命令见 [English README](README.md)。

## Docker 快速开始

Docker 与本地路径调用同一个 `relic` CLI。它不写入 API key、历史结果、cache 或 selected trace。

```bash
cp .env.example .env
mkdir -p outputs cache traces
docker compose build relic
docker compose run --rm relic check-env --scope core
docker compose run --rm relic smoke --mode mock
```

详见[环境与平台支持](docs/environment.md)：其中包含所有环境变量、runtime model / base URL、Docker volume、WSL2/PowerShell 使用方式、并发与内存策略。

## Inspector 与论文结果

```bash
uv run relic replay --trace /path/to/selected-trace.json
uv run relic inspect --trace /path/to/selected-trace.json
```

Inspector 默认只绑定 `127.0.0.1:8765`。它读取经过校验的公开 `relic-trace-v1`，不读取 checkpoint、private memory、provider messages 或 evaluator workspace。

可复制的正式入口、输出结构、transfer、CooperBench 边界以及当前不可补齐的外部资产，请从 [English README](README.md) 和 `docs/` 中的链接继续阅读。

## 许可

Relic 的原创源码以 [PolyForm Noncommercial License 1.0.0](LICENSE)
进行源码公开：遵守协议时，可以免费用于非商业目的，也可以修改和分发。
任何商业用途都需要事先取得 Hongyi Du 的单独书面授权；参见
[商业授权说明](COMMERCIAL_LICENSE.md)。

仓库中的第三方 benchmark 快照和其他第三方组件继续适用各自的许可证；
仓库级许可证不会替代这些条款。
