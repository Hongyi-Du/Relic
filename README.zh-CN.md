# Relic：从多智能体协作到持久组织能力

[![arXiv: 2609.32965](https://img.shields.io/badge/arXiv-2609.32965-b31b1b)](https://arxiv.org/abs/2609.32965)
[![项目主页](https://img.shields.io/badge/Website-Relic-24354b)](https://hongyidu.ai/relic/zh)
[![交互体验](https://img.shields.io/badge/Experience-Interactive-8b2942)](https://hongyidu.ai/relic/zh/experience)
[![文档](https://img.shields.io/badge/Docs-Guide-526c88)](docs/README.md)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776ab?logo=python&logoColor=white)](docs/installation.md)
[![CI 状态](https://img.shields.io/github/actions/workflow/status/Hongyi-Du/Relic/release-ci.yml?branch=main&event=push&label=CI)](https://github.com/Hongyi-Du/Relic/actions/workflows/release-ci.yml)
[![许可证：PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-6c5a7b)](LICENSE)

[论文](https://arxiv.org/abs/2609.32965) · [项目主页](https://hongyidu.ai/relic/zh) · [观看组织回放](https://hongyidu.ai/relic/zh/experience) · [文档](docs/README.md) · [English](README.md)

**当成员更替时，一个 AI 团队能留下什么？** Relic 把协作中反复出现的失败，
转化为由组织持有、可执行的协议。成员提出规则并共同治理；采用后的协议把
触发条件、责任分工、所需证据与执行后果接入后续工作。即使创建它的成员离开，
组织仍能保留、使用和修订这些协议。

![Relic 概览：协作摩擦形成经治理的协议；换了审核成员，同一规则仍然适用。](assets/readme/relic-overview.svg)

## 先看机制

共享接口发生变化，客户端因此报错，团队采用了一条接口审核协议。这条规则改变
运行时对行动的优先级和执行检查。审核成员更替后，审核要求仍然属于组织。

[![Relic 完整机制：共享失败、提案、协议采用、运行时绑定与跨成员留存。](assets/readme/mechanism-main.png)](https://hongyidu.ai/relic/zh)

可以在[项目主页](https://hongyidu.ai/relic/zh)探索交互主图，也可以阅读
[协议生命周期](docs/protocol_lifecycle.md)。实验依据见
[论文](https://arxiv.org/abs/2609.32965)与
[已报告的结果快照](artifacts/paper_results/paper_results.md)。

## 不安装，先看一局

[在线 Experience](https://hongyidu.ai/relic/zh/experience) 把组织动画回放与
Inspector 放在一起：先看协议如何影响工作，再查看对应的组织状态。

仓库也附带一个自包含的存档可视化。用 Python 启动静态服务器，
然后打开 `http://localhost:8000/`：

```bash
git clone https://github.com/Hongyi-Du/Relic.git relic
cd relic
python3 -m http.server 8000 --directory demo
```

选 **Story / 故事** 看协议生命周期的引导讲解，或拖动时间轴阅读整局。
仓库附带的 cattrs 回放是单局 parity run，并非论文汇总结果。
数据来源与操作说明见 [demo/README.md](demo/README.md)。

## 选择入口

| 你想做什么 | 从这里开始 |
| --- | --- |
| 理解研究 | [论文](https://arxiv.org/abs/2609.32965) · [交互主页](https://hongyidu.ai/relic/zh) |
| 复现实验 | **本仓库：** 冻结基准、B0–B3 配置、OrgEnv、评估器与[复现入口](reproduction/README.md) |
| 构建自己的智能体组织 | [**Relic-Agent**](https://github.com/Hongyi-Du/Relic-Agent)：与具体基准无关的可配置组织运行框架及 Inspector |

## 快速开始

使用 **Python 3.12+**、**uv** 和 **Linux / WSL2**。Windows 用户请把仓库放在
WSL 的 Linux 文件系统，例如 `~/relic`。如果已经按上面的命令克隆，
从 `uv sync` 开始即可。

```bash
git clone https://github.com/Hongyi-Du/Relic.git relic
cd relic
uv sync --extra dev --frozen
cp .env.example .env
uv run relic verify-benchmark
uv run relic check-env --scope core
uv run relic smoke --mode mock
```

这些检查不调用模型。Mock smoke 检查接口是否接通，不代表实验结果。
`uv run pytest -q` 运行发布测试集。[安装说明](docs/installation.md)介绍完整步骤；
[环境说明](docs/environment.md)包括 Docker、平台支持和输出目录。

## 运行论文实验

[configs/](configs/) 中的 YAML 与冻结的
[relic-main-v1 基准](benchmarks/relic-main-v1/)共同定义实验：
10 个任务 × 3 个随机种子 × 4 组条件 = **每个模型 120 个实验单元**。

<details>
<summary><strong>B0–B3 有什么区别</strong></summary>

| 条件 | 成员 | 决策方式 | 成员画像 / 能力条件化 | 协议生命周期 |
| --- | ---: | --- | --- | --- |
| B0 | 1 | 大模型直接选择行动 | 关闭 | 关闭 |
| B1 | 8 个持久角色 | 大模型直接选择行动 | 关闭 | 关闭 |
| B2 | 8 个持久角色 | SDL / 画像策略 | 开启 | 关闭 |
| B3 | 8 个持久角色 | SDL / 画像策略 | 开启 | 开启，包含运行时协议绑定 |

权威定义位于 [configs/arms/](configs/arms/)。
B2 与 B3 使用同样的结构化团队；B3 增加可治理、可执行的组织协议。

</details>

先生成一个不调用模型的执行计划：

```bash
uv run relic run-main \
  --model gpt-5.6-terra \
  --output-root outputs/main-study \
  --manifest outputs/main-study/source_main_manifest.json \
  --max-parallel 1 \
  --dry-run
```

在本地 `.env` 中填写兼容 OpenAI 的模型服务：

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-gateway.example/v1
RELIC_RUNTIME_MODEL=your-provider-deployment
```

然后运行一个保留 B0–B3 配对的实验批次：

```bash
uv run --env-file .env relic run-main \
  --manifest outputs/main-study/source_main_manifest.json \
  --resume \
  --max-parallel 1 \
  --workload w01 \
  --seed 1401
```

去掉 `--workload` 与 `--seed` 选择参数，运行该模型的完整实验。
各批次会写入 checkpoint、评估器元数据与 `experiment_runs.json/jsonl`，
顶层 manifest 链接这些结果。这些是你本地运行的结果，与仓库内论文结果快照分开。

直接执行 `uv run relic ...` 不会隐式读取 `.env`。调用模型时请使用
`uv run --env-file .env relic ...`。
`--model` 保留论文的模型标识；`--runtime-model` 或
`RELIC_RUNTIME_MODEL` 指定你的服务中的部署名称。

<details>
<summary>模型别名与断点续跑</summary>

凭证保留在本地 `.env` 中，不写入公开 trace、计划或 manifest。
`scripts/bash/` 中的轻量 Bash 包装器会自动加载允许的环境变量。

运行时模型名称的优先级为 `--runtime-model`、`RELIC_RUNTIME_MODEL`、
`ORG_LLM_RUNTIME_MODEL`、`OPENAI_MODEL`、`ORG_LLM_MODEL`。
Claude 分组没有通用覆盖时，可用 `RELIC_CLAUDE_OPUS_4_6_MODEL` 指定网关别名。
尚未执行的 dry plan 可在首次执行时补充部署名；执行开始后，resume 保留已记录的模型身份。
可选请求头与完整模型服务配置见 [environment.md](docs/environment.md)。

</details>

## 继续深入

| 主题 | 文档 |
| --- | --- |
| 主实验与输出 | [主结果复现](reproduction/main_results/README.md) · [配置](docs/configuration.md) |
| 新成员迁移 | [Transfer](docs/transfer.md) · [迁移复现](reproduction/transfer/README.md) |
| 公开 trace 与 Inspector | [Inspector](docs/inspector.md) |
| 评估器来源与容器 | [Evaluator](docs/evaluator.md) · [环境](docs/environment.md) |
| 人类席位扩展 | [`hci` 分支](https://github.com/Hongyi-Du/Relic/tree/hci) |
| CooperBench 扩展 | [`cooper` 分支](https://github.com/Hongyi-Du/Relic/tree/cooper) |
| 发布范围与可用资料 | [发布范围](docs/release-scope.md) · [范围与验证限制](docs/KNOWN_RELEASE_GAPS.md) |

主实验和迁移路径默认使用公开的 host evaluator。
可选 `--evaluator-bindings` 记录容器来源；
`--strict-reproducibility` 要求 digest 固定的 `linux/amd64` 镜像及资格校验哈希。

<details>
<summary>迁移实验命令</summary>

迁移实验增加新的 B2 Text 与 Exec 目标团队；Fresh 使用配对主实验中的 B2 结果。

```bash
uv run relic run-transfer --dry-run \
  --output-root outputs/transfer-v2 \
  --workload w01 \
  --seed 1401
uv run --env-file .env relic run-transfer \
  --manifest outputs/transfer-v2/transfer_manifest.json \
  --resume \
  --max-parallel 1 \
  --workload w01 \
  --seed 1401
```

</details>

<details>
<summary>旧版单元调度器的结果汇总</summary>

配对 `run-main` 已写入每批次的实验汇总。单独的旧版单元调度器使用
`run_manifest.json`；以下命令接收此格式，不接收 `source_main_manifest.json`：

```bash
uv run relic evaluate \
  --manifest outputs/user-run/run_manifest.json \
  --receipt-directory outputs/user-run/evaluation
uv run relic aggregate-user-runs \
  --evaluation-manifest outputs/user-run/evaluation/evaluation_manifest.json \
  --output-directory outputs/user-run/aggregate \
  --allow-partial
```

失败或降级的案例保留对应标签。本地汇总不会重新计算或替换论文报告的结果快照。

</details>

## 引用

使用本研究时，请引用[论文](https://arxiv.org/abs/2609.32965)。
BibTeX 见 [English README](README.md#citation)。

## 许可

Relic 的原创源码以 [PolyForm Noncommercial License 1.0.0](LICENSE)
进行源码公开：遵守协议时，可以用于非商业目的，也可以修改和分发。
任何商业用途都需要事先取得 Hongyi Du 的单独书面授权；参见
[商业授权说明](COMMERCIAL_LICENSE.md)。

仓库中的第三方 benchmark 快照和其他第三方组件继续适用各自的许可证；
仓库级许可证不会替代这些条款。
