# HCI Human Seat（P2 / P3）使用指南

**分支**：`hci`
**最后更新**：2026-09-17

## 1. 两个可独立体验的版本

P2 与 P3 都让真人持有 OrgEnv 中一个真实成员席位。真人拥有该席位与 P1/P2 完全相同的角色、权限、可见性和 action space；P3 不把人类缩成只做审批与行政的 observer。代码修改、测试、review、文档、实验、会议、任务与治理动作都仍可做，只是 P3 把这些能力统一收进 **Human–Organization Liaison** 这一条自然语言入口。

- **P2 · Transparent Liaison**：liaison 是主要输入，seat-visible 的组织面、成员、workstreams 和对象仍默认透明。
- **P3 · Integrated Liaison**：界面是一个删去无关表面的 Codex-like task thread，固定使用 Victor 的真实席位，不再显示 seat picker。项目状态、对话、确认、执行反馈和事件摘要都在同一线程；底层组织默认隐藏，但 `Show organization` 永远可达。

Liaison 解释（interpret）、总结（summarize）、澄清（clarify）、委派（delegate）和路由（route），但不自行治理。它可以把真人确认过的开发或测试请求交给组织中的 agent，也可以建立 Victor 私有的 execution department：多个具有独立模型上下文的 workers 代表 Victor 完成代码、测试、review 与 integration workflow。Workers 不是 organization member；每个共享动作仍作为 Victor 的普通 P1/P2 action 进入 gateway。Liaison 不能在真人未确认时替人审批、分配、改优先级或改协议，也不能把请求变成 P1/P2 action space 之外的新权力。

真人、agent 和 liaison 确认后的动作仍走同一个 `human.gateway` 与 `OrgWorld._execute_action_pipeline`。组织内看不到“这是人类发的”标记，也没有人类专属权重；研究侧只在 `controller_log` 与 HCI replay 里保留控制来源和 `execution_mode`。

P2 的可见性边界是席位边界：组织简报、liaison 工具和对象抽屉都只消费该席位已经过滤过的 `seat_view`，不能读取同事的 private task、private document、branch、sandbox 或 agent interior。

## 2. 启动

```powershell
cd relic
python -m pip install -r requirements-hci.txt
python tools/run_hci.py --paused
```

这份轻量依赖已包括 OpenAI-compatible SDK；默认启动不请求模型。启用真实模型时，配置对应网关和密钥，再用 `--llm` 启动。一次页面可用或一次只读回复不等于委派全链路通过。

打开两个版本：

```text
P2  http://localhost:8100/org/seat
P3  http://localhost:8100/org/liaison
```

如修改前端，先构建提交到仓库的静态产物：

```powershell
cd environments/org_env/frontend/app
npm install
npm run build
```

多人体验可用 `python tools/run_hci.py --host 0.0.0.0`，但 per-seat token 是当前唯一凭据，只应在可信网络开放。

## 3. 进入席位

P2 首次打开会看到可认领成员；选择后，真人获得该成员的角色、权限和可见范围，同一席位同时只能有一个持有者。

P3 没有选择环节：打开 `/org/liaison` 后，服务端只会创建或恢复 Victor 的席位。浏览器不能提交其它 `agent_id`；即使本地残留了另一名成员的 P1/P2 token，P3 也会在服务端拒绝该 token，再请求 Victor 会话。Codex 内嵌页、Chrome 等不同客户端的 `localStorage` 不共享，因此无 token 的后续 P3 客户端会附着到已经存在的同一个 Victor 固定会话，而不是再次 claim；它们看到同一条秘书线程和同一份确认状态。P3 因而固定继承 Victor 在 P1/P2 中原有的角色、权限、可见性和完整 action space，而不是获得一个额外的秘书角色。

默认热身后的组织已经有任务、消息、PR 和提案，便于直接观察。若要研究某个固定时点，应使用 `--paused`，先认领席位、记录状态，再手动恢复时钟。

## 4. P2 界面

P2 是两个并列面板：

```text
┌─────────────────────────────────────────────────────────────┐
│ LanternForge · transparent liaison · clock · seat           │
├──────────────────────────┬──────────────────────────────────┤
│ Transparent organization │ Human–Organization Liaison       │
│                          │                                  │
│ Organization Brief       │ natural-language question       │
│ workstreams / blockers   │ answer + visible evidence       │
│ agents / my work         │                                  │
│ mentions / decisions     │ consequential intent → draft    │
│ visible objects          │ [Confirm and route] [Discard]    │
└──────────────────────────┴──────────────────────────────────┘
```

左侧保持组织透明：先显示 `Organization Brief`，再显示可见 workstream、agent roster、你的任务和需要你处理的事项。点击带对象引用的条目可打开 `ObjectDrawer`，查看该席位可见字段和当前允许的动作。

右侧 liaison 是默认自然语言入口。广义状态问题（例如“组织现在在做什么”“谁负责当前 blocker”）从当前 `organization_brief` 回答，并展示可展开的 evidence。P2 的 standalone working agent 仍可在没有 LLM 时提供有限的 seat-visible 离线工具答案；P3 的所有真人输入必须先由模型做语义路由，provider 不可用时明确报错，绝不用 regex/keyword/template 假装理解。

## 5. P3 集成式 Liaison 界面

P3 打开后直接进入 Victor 的 liaison-first 桌面工作区，不经过 seat picker。界面结构采用「左侧项目/会话导航 + 中央秘书线程 + 右上角按需详情」：中央线程承担 FYI、需要人类判断的建议、真人消息、秘书回复、执行团队进度与完成反馈；底部始终只有一个自然语言输入框。传统 IDE 的文件树、终端、diff、模型选择器或 agent manager 不会常驻并挤占这条主交互，但 source visibility 没有被删掉：需要核查原始材料时，Secretary 会在右侧打开一个小型 read-only IDE。组织报表同样默认不常驻，右上角详情按钮才打开当前对话的工作、参与者、原因、证据与更完整组织状态。

```text
┌──────────────┬─────────────────────────────────────┬───────────────┐
│ LanternForge │ current work / conversation     [☷] │ In this      │
│ + New        ├─────────────────────────────────────┤ conversation │
│ Conversations│ FYI · organization handled it       │ current work │
│ Workspace    │ Decision needed · human judgment    │ involved     │
│ summaries    │ human message → secretary reply     │ why / status │
│ feedback     │ execution team working / ready      │ evidence     │
│ protocol     │                                     │              │
│ Victor seat  │ [one composer................... ↑] │ org details  │
└──────────────┴─────────────────────────────────────┴───────────────┘
```

右侧栏默认关闭。右上角的 details/sliders 按钮只控制 progressive disclosure，不改变世界；首次打开优先给出 `In this conversation`，包括当前 workstream 及状态、谁参与、为什么需要这项工作或人类判断、相关 evidence。只有真人继续点 `View organization details`，才显示同一 Victor `seat_view` 范围内的其它 workstreams、agents、decision inbox 与 recent activity。

### 5.1 Secretary 集成的 Resource Inspector

Secretary 默认给摘要，但摘要中的 review、测试、文件或代码结论都可以继续回到原始材料。点击 event/review 的 `Open source`，或直接说“打开这个 PR 的原始 review 和 diff”“给我看失败 CI 对应的代码”“展开这份 proposal 的原始证据”，模型会返回结构化 `open_resource`，页面在同一个右侧 slot 打开通用 Resource Inspector。它与 Slack thread 和 `In this conversation` 面板互斥，不会让左侧导航或中央对话跟着长 diff 一起滚动。

FYI 的 `Show details` 不是重复标题：它显示该事件在 Victor P2 feed 中已经可见的完整结构化记录，并列出 event 明确关联、当前仍可见的 task、PR、proposal、meeting 或 release object。`Show trace` 再展示 event 当时的事实、当前对象状态和来源链。只要有 current source，卡片会给出明显的 `Open full context`，在同一右侧 Inspector 展开完整对象；没有 source、source 已离开可见范围或 domain 没有保存原文时，会明确说明原因，而不是补写一段看似完整的解释。

Decisions 页面中的每张卡必须说明 **Matter / Why this is surfaced / Impact**，列出可用的自然语言决策方向，并提供 `Open decision context`。proposal 会打开完整 review packet（problem、solution、required actions/participants、benefits、costs、risks、failure modes、scores 和 approval state）；PR 会打开真实 review/CI/diff/code；meeting 会打开完整 agenda、participants、当前 RSVP 与公开 outcomes。决策 option 只会把指令放进同一个 composer，真人仍可修改、追问或不发送；任何批准、拒绝、review、参会或缺席都不会因为打开详情而自动发生。

PR 是第一种完整 adapter，面板提供 Summary / Review / Files / Diff / Tests / Evidence：reviewer、approve/request-changes verdict、comment、CI run/check/failure、commit/patch lineage、stored exact `unified_diff` 和 patch/file code 都来自当前 world 的真实 domain record。`change_summary`、`pseudo_diff` 和 Secretary prose 会明确标成 summary，不会伪装成 exact source。底层没有保存某条 comment、test stdout、diff 或 file snapshot 时，对应 tab 直接说明 `unavailable` 的原因；不会为了填满界面自动生成一份。

同一 Inspector contract 也承载 task、document/artifact、proposal/protocol、experiment/result、meeting、release candidate 等 review-like object。它们先展示当前 P2-visible raw projection 与 provenance，存在真实 section 时再展开；不支持或已经不再可取的 section 保持诚实空态。Browser 只能使用服务端签发、绑定 Victor token 的 `ri_...` handle；每次 refresh 都重新校验当前 seat visibility、PR→commit→patch→artifact parent chain 和 source revision。拥有某个 action 不等于有权读取某个 source；hidden tests、reference repo、evaluator answer、controller log、private reflection/memory/wish 与其他成员 private artifact 始终不可见。

Inspector 本身只读，不需要确认，也不会改变组织。若真人在看完材料后要求 approve、request changes、run CI、edit 或 merge，那些请求仍从唯一 composer 进入 model compiler，形成普通 draft 或 execution job，再经过原有确认和 current-state gateway。

秘书提出澄清问题时不会弹出必须立刻回答的 modal，也不会锁住其它对话。真人可以先反问“为什么需要这个信息”、继续查看 evidence/trace、补充相关背景，或稍后再回答；在上下文足够之前，liaison 保持 `clarification` 状态且不产生组织动作。

### 5.1 会前 RSVP 与秘书代为传达

当 agent 安排的会议包含 Victor 时，P3 会在会议启动前插入独立的 `Decision needed / Meeting invitation`。卡片列出公开 agenda、participants 和当前状态，但不会给出要求立即点击的固定选项；真人仍用底部 composer 自然语言回答，也可以先连续追问会议为什么召开、谁参加、现有证据是什么。只要 Victor 的 claimed human seat 还没有确认 RSVP，该会议保持 `scheduled`；其它 agent、其它会议和普通组织工作继续运行。

可用的自然语言语义包括：

- “我参加” → 准备真实 `attend_meeting` draft；
- “我不去，你替我表达证据不足，回来告诉我 CI 和最终决定” → 先准备一条只含该原话的 meeting-room message，再准备 `skip_meeting`，同时在 HCI-local plan 中记录会后关注点；
- “我不去，不用表达，只告诉我最后决定” → 不生成立场，记录 return focus，再准备 `skip_meeting`；
- “我不去，也不用转达或带回，两个都不要” → 只准备 `skip_meeting`；
- “你去看看” → 信息不足，秘书继续追问，不猜测立场或回传范围。

这里“秘书去”的组织语义不是创建一个 Secretary agent，也不是让秘书获得席位权限。秘书只能把 Victor 已明确给出的原话，经 Victor 的普通 `send_message` action 发到该 meeting room；组织内 sender 仍是 Victor，HCI 研究日志记录 `liaison_assisted`。若同时包含观点和缺席，P3 只显示一个可读的 meeting plan 和一次确认；组件 `send_message` / `skip_meeting` draft 不暴露给真人，服务端先发送已确认原话、再记录缺席，避免会议先启动而观点还未送达。

真人确认 attend 后，会议启动时 Victor 才会进入 attendee/busy 状态；确认 skip 后，会议可以由其它参与者开始，但 Victor 不会被计为 attendee，也不会被自动安排 notes、summary 或 action item。会议结束后，秘书只从 Victor 可见的 meeting note、decision、action item、unresolved question 和 room message 形成 `Meeting report`。如果真人明确说无需带回信息，线程只给出会议已结束的最小确认；如果没有公开记录，秘书会说记录不足，不能用 agent reflection 或 memory 补写结论。

### 5.2 消息筛选与 human-related event 路由

默认 `secretary_triage` 下，秘书先读取 Victor 的 filtered P2 message surface，只立即转达明确指向 Victor 且属于 `urgent|incident|blocker|decision_relevant|policy_relevant` 的消息；普通 channel traffic 不会重新淹没主线程。右侧 `In this conversation` 显示当前 message delivery policy。真人可以直接说“以后所有消息都给我看”或 “show me all messages” 切换到 `all_messages`，此后 Victor 可见的每条消息都从秘书线程传达；说“恢复筛选，只告诉我重要消息”即可切回。这个选择不扩大 channel/DM 可见性，也不会把秘书变成组织成员。

明确把 Victor 列为 actor、participant、owner、assignee、target、reviewer 或 approver 的有效 seat-visible event 也统一经秘书传达。P3 为此保留一个有界、按 seat visibility 过滤的 HCI delivery journal：即使浏览器短期关闭、P2 的 120 条 working feed 已滚动，重新打开后仍可补送尚未呈现的 human-related event；世界 reset 或 seat release 会清空 journal，旧组织的数据不会进入新会话。

每张 event、directed message、meeting invitation/report 卡现在都是可回复的 thread root。点击 `Reply` 后，Secretary 的模型会同时收到该卡的完整 public summary、该 thread 的后续消息、当前 `team_progress` 和 `recent_activity`，所以“这是什么”“这是我的 PR 吗”“现在谁在负责”不再脱离卡片重新猜。`team_progress` 按成员分别列出 active/completed task、自己 authored 的 PR、review assignment 与实际 approval；PR 的 owner 永远取 `author`，reviewer 不会因为触发了 `pr_reviewed` event 而被显示成作者。历史 event 的语义状态也不会随着 PR 后来 merged 而变成第二张重复卡；详细信息可以同时说明事件当时发生了什么和对象当前是什么状态。

P3 的能力等价由后端契约保证：liaison 的 action catalog 在每次推理时直接由 canonical `ORG_ACTION_CATEGORIES` 与 P2 `affordances` 生成，不维护一份会漂移的手写子集。当前 registry 的 189 个 action type 全部进入 P1/P2/P3；`run_ci` 同时保留 PR 与 branch 两种 target form，因此共有 190 个参数 schema。代码修改、测试、task/PR/review、document、experiment/result、meeting、proposal/protocol、release、message、issue、artifact、search、external bridge、payroll/hiring 与 time 的每个原始 verb 都进入同一个 natural-language compiler；一个请求可成为有序的 multi-action draft。122 个 verb 到达其专用 runtime handler，另外 67 个 legacy verb 保持普通 agent 已有的 shared generic registry-event 语义；秘书不会把后者伪装成具有不存在的专用副作用。新增 action 未进入 catalog、compiler、execution-worker validator 或 gateway route 时，全量 parity 测试会失败。编译器不能把明确的 edit/test/approve/propose/assign 请求降级为泛化 `send_message`。秘书列出任务/PR 等对象时会同时保存经当前 Victor 可见性校验的结构化 ID；所以后续“第一个”“这个任务”不再靠解析旧回复的字面文本。若模型已经选对 action 但漏了 ID，系统只允许模型利用这份 reference set、当前 task focus 和真实可操作候选修复一次自己的 JSON；后端不使用 regex 从真人原句补猜。仍无法唯一确定时整批 draft 都不创建，只产生带可见候选项的普通澄清。真人确认后仍通过同一个 gateway；是否允许最终仍由席位角色、目标可见性和当前组织状态决定。

任务分配中的“谁做什么你来决定”是一项明确、受限的选择委托，不是信息缺失。`team_progress` 会把每个未分配、未结束任务的精确 ID、标题、状态和优先级交给模型；compiler 同时提供可选成员的角色、在线状态与当前工作负载。通用 compiler 已识别 `assign_task_owner` 却漏填批量目标时，会调用一个窄 schema 的分配规划模型再次判断真人是否明确委托：只有判断为明确委托才可据此选择负责人、覆盖全部 requested/eligible task，并遵守“给 Victor 留一个”等数量约束；否则零草稿并请求明确授权或负责人。结果仍是逐项可审查的 `assign_task_owner` 草稿，确认前不改变组织。批量草稿缺同一种参数时只显示一次说明，不重复刷屏。

任务交接的最低可用闭环如下。认领和启动 execution team 是两个清楚的确认边界，中间的代码阅读与报告均为只读：

```text
“哪些任务没认领？” → Secretary 返回编号列表 + 已校验对象引用
“我认领第一个”     → pending pick_task → Victor 确认 → 当前任务焦点
“读代码并告诉我缺什么，再交给 Walker”
                     → read task/repo/source/tests
                     → Markdown 缺口报告 + pending Worker plan
“Start execution team”
                     → Worker audit + P1/P2 gateway actions
                     → Secretary 回传真实结果；Worker 回到 dormant，可继续复用
```

Secretary 回复支持标题、列表、强调、行内代码和 fenced code block；链接只接受 HTTP(S)，页面不用 raw HTML 渲染。Markdown 只负责可读性，代码块不会自动执行其中的任何 action。

为使自然语言真的能够落到结构动作，compiler context 同时携带当前席位可见的 member、channel 和完整 object symbol table：task、PR、branch、document、experiment、result、meeting、proposal、protocol、release candidate、release、message 与 issue；还包含 bounded recent dialogue、organization brief 和可见 repo path。该上下文不读取 private cognition，也不把隐藏对象暴露给模型。模型返回的整批动作先按 live schema 做 action type、字段白名单、必填参数与重复项校验，全部通过后才显示确认卡。

纯只读的状态、归属和 event 解释仍然先调用模型，但 router 可以直接基于上述结构化 grounding 返回 `grounded_answer`，不再为了回答“其他人进度怎样”额外启动一个没有 event context 的 24-step worker loop。凡是要求检查代码、运行测试、review、写实现、合并、审批、提案、委派或其它真实工作的指令仍进入完整 compiler / execution department，不能被 `grounded_answer` 吞掉。

agent 或组织产生的新状态会作为摘要回到同一线程。每条摘要都可按 `Summary → Evidence → Organizational trace` 展开。文案严格区分：

- `requested`：请求已送达，不代表任何 agent 已接受；
- `assigned`：owner 或 follow-up task 已在组织记录中出现，不代表已开工；
- `active`：可见状态表明确有工作、review、测试或会议活动；
- `verified`：某次明确的可见测试/CI 通过，不等于整个任务完成；
- `completed`：只在相应对象的权威终态出现时使用。产品 task 只有 `merged`、`released` 或 `done` 才可作为完成反馈。

事件线程不是 raw feed 的逐行镜像。`tick`、单纯 `read_feed`、空 wish clustering、无结果的 background job 和没有安全可述 actor/action/object/result 的内部事件不会进入主线程。P1/P2/P3 的 `human_project_workspace` world 从 source boundary 禁用 scheduled funding tranche 和 synthetic customer/churn loop，所以 HCI 页面不会出现与项目协作无关的融资、客户注入或自动流失任务；普通组织实验仍保留这些动力学。相邻的有效事件合成一个紧凑 `Organization update`，每项保留组织 tick、具体成员、动作、可见对象、状态或结果；同一对象在初次加载时只显示最新状态，不重放 `created → under review → approved` 的每个中间事件。一次 release readiness check 派生的 blocker issue/task burst 会合并成一条摘要，直接给出 gate 数量、follow-up owner 分布，并在 evidence 中列出可见 gate、task 与 owner。`Show details` 必须提供比标题更具体的完整 P2-visible event record，`Show trace` 再给出当前关联对象与来源链，不能只重复“发生了一个事件”。

meeting、decision、proposal、protocol 变化都会各自生成有边界的事件摘要。agent 的 private reflection、memory、wish 与 policy trace 仍不可见；若一次反思产生了公开 proposal、protocol、meeting decision 或 task/PR 变化，P3 汇报这个公开产物及来源，而不伪装成能够读取 agent 内心。

### 5.3 Victor 的 execution department

当真人要求“你去完成”“开一个开发/测试/review team”“检查后给我报告，没问题再合并”这类 substantive workflow 时，Secretary 会先读取当前 task/object 与相关 source/test evidence，再可在同一轮返回结构化 Markdown review 和一个 `Execution team` job。这个 job 不是进度占位图：在后端实际存在 job state、稳定 worker id、独立 worker model calls、action result、evidence timeline 和 final report；报告本身不等于启动，仍须真人点击 `Start execution team`。

流程如下：

```text
human natural-language goal
  → Secretary model creates goal + completion criteria + worker assignments
  → Execution team card (pending confirmation; zero world mutation)
  → Start execution team (one semantic confirmation)
  → worker model loop: inspect → choose tool/action → semantic scope audit
  → P1/P2 gateway as Victor → real OrgWorld result/evidence
  → next worker receives earlier evidence/report
  → Secretary final evidence-based summary in the original main/thread conversation
```

每个 worker 都是一次真实、独立的模型工作上下文，而不是初始模型预写 actions 后附加的 metadata。常用 roles 是 researcher、developer、tester、reviewer 和 integrator；所有 role 都从同一完整 P1/P2 action schema catalog 选择动作，role 名称本身不产生任何额外 authority。UI 显示每个 worker 的 assignment、状态、model call 数、最新 action/evidence 与 report；如果 worker 只是排队、action 被 gateway 拒绝、测试失败或缺少 PR，页面必须显示对应事实，不能显示 ready/completed。

Worker 不是一次性对象。Secretary 为 Victor 维护一个 persistent execution-agent roster：每个 agent 有稳定 `worker_id`、专业角色、累计 activation/model-call 计数，以及带 `job_id/run_id/runtime_epoch` 来源的历史报告与 evidence summary。一个 job 结束后 agent 从 `active` 回到 `inactive`（UI 显示 dormant），不会被删除；同一任务的修改、复查或继续实现应由 Secretary 模型从 `execution_agents` 中选择原 worker_id 再激活，而不是每轮建立陌生 agent。只有没有合适 inactive agent 时才创建新人。

持久身份和单次执行严格分离：每次激活都有新的 `run_id` 与 generation，历史 job 保留自己的 assignment/status/model-call/report 快照，后来的复用不能改写旧卡片。一个 persistent agent 同时最多属于一个 running job；取消、seat release 或 runtime 更换后，迟到的模型结果必须通过 `(runtime_epoch, worker_id, run_id, generation)` fencing 丢弃，不能进入 gateway。长期记忆只帮助定位和延续工作，不继承旧授权；每次动作仍重新读取 live seat context 并走当前 gateway。

这里的“持久”边界是同一个组织 runtime：agent 可以跨多轮任务、返工和 seat 暂时释放而休眠/复用；切换 pack、恢复成另一个 world 或重启服务会开始新的组织生命周期，不会把旧上下文暗中带进新组织。

`Start execution team` 只确认这一个 goal 与 worker assignments，不是允许 workers 随意使用 Victor 身份。每个 consequential step 都先由独立 scope-audit 模型核对它是否直接属于真人确认的目标，再在当前 world 上经过 role、visibility、target、context 和 protocol gate。`review/test，没问题再 merge` 中的 merge 必须看到匹配目标 PR 的真实 review/CI evidence；否则 integrator 只能报告 blocked。`run_ci(branch_id)` 只寻找该 branch 的 open PR，不会回退到另一个 PR。

真人可在 job 运行时继续使用 Secretary，并可点击 `Cancel execution` 发出 cooperative cancellation；已经进入 gateway 的单个原子 action 不回滚，后续步骤会停止。Job settled 后 Secretary 自动回传 final summary。Summary 明确区分 action succeeded、tests passed、PR opened/approved/merged 和 task completed；其中任何单项都不能替代其它证据。

### 5.4 Secretary 的长任务上下文管理

Secretary 的一次自然语言请求不是“最近若干条消息 + 一个固定 step 数”的无状态循环，而是一个 request-local working context。这个 context 借鉴 Codex 长任务中可公开验证的上下文管理思路，但不声称复制其私有实现：原始请求与 thread id 是不可压缩的 **task anchor**；每次 read-only tool 的结果进入带 source、参数、step 和 fingerprint 的 **evidence ledger**；已完成的精确 `tool + args` 进入 **call ledger**，同一读取不会再次执行。

当证据总量增大时，系统按来源压缩每条 observation 的正文，保留 source identity、fingerprint、开头与结尾，并明确记录省略的字符数。它不会只保留“最近 8 步”，也不会因为压缩而丢掉早期的 task/PR/代码证据。普通 conversation history 与本次工具证据分开：recent dialogue 不再重复携带 tool output，模型每轮都重新看到同一个 task anchor、当前完整 source index、已做过的读取和剩余预算。

`MAX_STEPS=24` 是异常循环的硬上限，不是正常停止点。若模型重复同一读取、证据不再变化，或只剩综合预算，context 进入 `synthesis_only`：工具集合变为空，模型必须用已有证据返回可读答复、澄清、结构化 draft 或 execution plan。UI 在等待期间显示后端真实的阶段、步数、已完成工具数、独立证据源数和重复读取拦截数；这些是执行状态，不包含模型 hidden reasoning。若最终模型仍拒绝综合，Secretary 明确报告模型没有得出结论，且不会伪造已完成动作。

Execution department 的每个 worker 也使用有界长任务上下文，而不再只有 6 个决定：最多 24 个有效步骤、48 次总模型调用，并为最后的 tool-free report 与 completion audit 固定预留两次调用。每轮都重复不可压缩的 `task_anchor`（human request、goal、completion criteria、assignment）；精确相同的 read tool + 有效参数会在执行前去重且不消耗有效步骤。完整来源保存在 private evidence ledger，模型只接收有界 excerpt；最近 10 条详细 history 之外的记录压成结构摘要，跨 worker timeline 保留早期事件计数与最近 32 条。用满工作预算仍必须先综合再接受/拒绝 completion，不能再次以“达到步数上限”冒充任务结论。

Codex webview 与 Chrome 可以同时连接同一个固定 Victor session。若上一条请求仍在调用模型，下一条消息会先出现在共享 transcript 中并显示 queued 数量，然后按到达顺序继续；后端不再返回 `agent_busy`，也不会让 queued turn 提前污染当前请求的 context。每一条排队的人类指令仍会独立经过模型语义路由。

FastAPI 的 `/liaison/ask`、confirmation、meeting-plan confirmation 与 release 都把同步模型/执行工作移出 ASGI event loop。这样一个标签页正在等待 semantic router、测试或 world handler 时，另一个标签页的 state polling、pause/resume 与新消息入队仍可响应；浏览器看到的 working/queue/job 状态始终来自同一个后端 session。

Semantic router 每次还会收到与 UI 同源的 `project_context` 与 `runtime_context`：当前 product/company/stage，以及 live pack、world tick、running/paused、engine、event source 和 model attachment。它不能在页面明明显示项目与模拟状态时声称“上下文没有这些字段”，也不能凭对话历史猜这些事实。

如果 provider 在语义路由时返回临时错误，服务器仍把已经收到的人类原文保存在同一 main/thread 中，并追加一条 `model_error`：明确说明本次模型没有完成解释、没有 regex/template fallback、没有生成或执行动作，可以重试。这种情况不再显示成 `Not sent`；只有网络请求根本未到达服务器、token 无效或 thread root 无效时才保留真正的发送失败状态。

主 composer 在 Conversation、Decisions、Activity 任一过滤页都可用；一旦发送，UI 会先切回 Conversation 再插入 optimistic human turn，因此消息不会被当前过滤器隐藏成只剩 `Working…`。这条 turn 随后由 server transcript 确认，第二个浏览器标签页会看到同一内容。

## 6. 后果性请求、澄清与确认

任何可能改变共享状态的请求都不能由 liaison 在未确认时直接执行。单个或显式 ordered action 使用 draft；substantive multi-worker workflow 使用 execution job。两条路径固定为：

```text
human request / follow-up context
  → liaison interpretation or clarification
  → optional read-only questions and evidence inspection
  → DraftAction (pending, no world mutation)
  → human reviews exact action + params + rationale
  → Confirm and route
  → authenticate current seat token
  → re-run role / visibility / target / protocol gates on current world state
  → shared action pipeline

multi-worker request
  → model-generated goal / completion criteria / worker assignments
  → ExecutionJob (pending_confirmation, no world mutation)
  → human starts the team once
  → each worker independently selects and audits one step
  → current-state gateway as Victor (`liaison_subagent`)
  → evidence-bearing final Secretary report
```

点击 `Discard draft` 不产生组织动作。确认也不等于一定成功：如果确认时角色、对象可见性或协议门已经变化，gateway 会拒绝并保留可解释错误。成功的动作以 `execution_mode="liaison_assisted"` 写入研究侧 `controller_log`，但组织内部仍只看到该成员做了一个普通动作。

P2 standalone working agent 的离线模式仍必须保留安全边界；它只能提供有限只读工具与既有测试 fallback。P3 不提供人类语义的离线 parser：没有 model 或结构化调用失败时不生成 action、meeting plan 或 execution job，只返回显式 provider/model error。在真人确认 draft 或 execution job 之前，`action_log`、消息流和世界状态都不改变。

## 7. 组织简报与来源

`organization_brief` 是 `seat_view` 的纯投影，不是新的权威状态。它包含：

- `what_is_happening`：当前可见工作、状态、原因、负责人和 refs；
- `needs_your_decision`：只列该席位本人等待处理的 review、proposal 或 meeting；
- `workstreams`、`blockers`、`agents`：给界面使用的同源别名；
- `source_refs`：可追溯的 `{kind, id, title}` 来源。

接口：

```text
GET /api/org/human/view?token=...
GET /api/org/human/brief?token=...
POST /api/org/human/agent/send
GET /api/org/human/agent?token=...
POST /api/org/human/agent/confirm
POST /api/org/human/agent/discard

POST /api/org/liaison/session
GET /api/org/liaison/state?token=...
POST /api/org/liaison/ask
POST /api/org/liaison/prepare
POST /api/org/liaison/confirm
POST /api/org/liaison/discard
POST /api/org/liaison/execution/start
POST /api/org/liaison/execution/cancel
GET /api/org/liaison/evidence?token=...&ref=...
GET /api/org/liaison/trace?token=...&ref=...
GET /api/org/liaison/resource?token=...&ref=...
GET /api/org/liaison/organization?token=...
```

所有 seat-private 读取和写入都要求 token。P3 的 `/session` 不接受 `agent_id`，但可把多个本地 P3 客户端连接到同一个 Victor token；其余 P3 接口还会再次验证 token 确实属于 Victor。因多个页面共享同一身份、对话与 pending drafts，只应在可信本机或可信网络开放。

## 8. 时钟与日志

顶部 `Pause the team` / `Resume the team` 控制同一个 OrgWorld 时钟。暂停只停止自治 agent 的推进，不妨碍真人阅读或提交已确认动作。

HCI replay 记录：

- human → liaison 请求；
- liaison 回复和只读 tool output；
- draft 创建、拒绝、确认和丢弃；
- 确认后 action id、成功状态与对象变更。
- P3 clarification、delegation/request 状态和 visible object state change；
- P3 meeting invitation、自然语言 meeting plan、真实 RSVP、message delivery preference 与会后 return brief；
- P3 execution job 的计划、start/cancel、每个 worker model call、scope audit、gateway action/result、worker report 与 final Secretary summary；
- 被 delivery journal 补送的 Victor-related visible event（仍不含其它席位的 private state）；
- 完成、会议、决定及其它可见事件的摘要与 evidence/trace disclosure。
- model-routed 或点击触发的 Resource Inspector open/refresh，以及 opaque resource kind/section/provenance；原始 code/diff 本文不复制进普通 activity feed。

这些日志用于 HCI 条件分析，不作为组织内部信号，也不证明 evaluator 结果或科学效应。

## 9. 验证

```powershell
.\.venv-hci\Scripts\python.exe -m pytest -q tests/org_env/test_human_*.py
cd environments/org_env/frontend/app
npm run typecheck
npm run build
```

核心验收：P3 与 P1/P2 action capability set 相等但没有 authority multiplier；每个 execution worker 有独立真实 model call，确认前零 action，启动后只通过 Victor gateway，target identity 不漂移，失败/取消/final report 与真实 evidence 一致；澄清不会强迫即时回答或提前行动；未 RSVP 的 human meeting 不会 auto-attend，且只 hold 该 meeting；confirmed skip 不 busy、confirmed attend 才进入 meeting；human attendee 不运行 autonomous meeting sub-action；秘书只转达 human 确认的原话；`secretary_triage ↔ all_messages` 可用自然语言切换且不扩大可见性；会后只依据公开记录回传；状态问答不依赖 agent 自愿回复；草稿不改变世界；确认时重新通过 gateway；assignment/activity/completion 不混写；P3 provider 失败明确失败而不用 regex/template 假装理解；任一 review summary 可打开同一 seat-visible 的真实 source/review/CI/diff/code，缺失明确、不可见拒绝、跨 token handle 无效，Secretary 自然语言与点击入口都使用同一通用 Inspector；组织世界里没有 human-origin 标记或 5× priority；`dist/seat.html` 与 `dist/liaison.html` 均与源代码同步。

## 10. 调试

全知 inspector 在 `http://localhost:8100/org/inspector`。它只用于研究和可见性对照，不是 P2 参与界面。若 token 因世界 reset 失效，退出重进；必要时在浏览器控制台执行：

```javascript
localStorage.removeItem("socio.seat.token");
location.reload();
```

本公开指南是 P2/P3 的当前操作与行为契约说明。运行与平台边界见
[环境与平台支持](environment.md)，公开观察/回放的隐私边界见
[Relic Inspector](inspector.md)，源代码来源与受控适配范围见
[Source provenance](source-provenance.md)。开发期的私有设计和计划文档不属于
release 文档集，也不作为公开使用前提。
