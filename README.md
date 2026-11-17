# Resumable Training API（可恢复单作业训练服务）

FastAPI + CPU 版 PyTorch 实现的**单作业训练服务**。训练工程师在作业被抢占（暂停、
进程强退、容器被杀）后，可以从最近成功的检查点继续训练；在**同环境、同种子、同
配置**下，恢复后的逐批损失与最终参数与**不中断训练逐位一致**。

- 模型：两层全连接网络 `Linear → ReLU → Dropout → Linear`
- 优化器：Adam；每轮（epoch）打乱全部样本后按批训练
- 数据：仓库内置固定分类数据集 `data/classification.csv`（600 样本 / 2 特征 / 3 类）
- 默认配置：`config/default.json`（可在启动时按字段覆盖）

## 目录结构

```
app/
  main.py        # FastAPI 入口（/docs 自动生成）
  trainer.py     # 单作业状态机 + 确定性训练循环 + 暂停/续训
  checkpoint.py  # 原子检查点读写（tmp + fsync + rename）
  config.py      # 默认配置加载与校验
  dataset.py     # 固定 CSV 数据集加载
  model.py       # 两层 FC + Dropout
config/default.json      # 默认训练配置
data/classification.csv  # 固定分类数据集（scripts/generate_data.py 生成）
scripts/generate_data.py # 数据集生成脚本（固定种子，仅作溯源）
scripts/verify_repro.py  # 端到端验证：不中断对照 / 暂停续训 / 强退恢复 等
Dockerfile, docker-compose.yml, requirements.txt
```

## 快速开始

### Docker Compose（推荐）

```bash
docker compose up --build
# 服务: http://localhost:8000  交互式文档: http://localhost:8000/docs
```

检查点保存在命名卷 `trainer-runs` 中，容器被杀或重启后数据不丢。

### 本地运行（开发）

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 环境变量: RUNS_DIR(默认 runs/)  CONFIG_PATH(默认 config/default.json)
```

## API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务信息 |
| GET | `/config` | 默认配置与运行时信息 |
| POST | `/train/start` | 启动新训练。Body: `{"force": false, "config": {覆盖字段}}` |
| POST | `/train/pause` | 暂停。**训练停止且完整检查点持久化后才返回 200** |
| POST | `/train/resume` | 从暂停态/最近检查点续训（并发只放行一个训练者） |
| GET | `/train/status` | 状态与进度（epoch/batch/百分比/检查点健康度） |
| GET | `/train/losses` | 逐批损失（支持 `since_step`、`limit`） |
| GET | `/train/parameters` | 下载当前模型参数（.pt，取自最近持久化检查点） |
| GET | `/train/checkpoint` | 下载完整检查点（模型+优化器+RNG+历史） |

交互式文档（Swagger UI）在 `/docs`，可直接在浏览器里调通上述接口。

### curl 示例

```bash
# 启动（覆盖部分配置；已有运行需 force=true 或先续训）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8, "batch_size": 32, "lr": 0.01}}'

# 查询进度
curl -s localhost:8000/train/status | jq '.state, .completed_batches, .total_batches'

# 逐批损失
curl -s 'localhost:8000/train/losses?limit=5' | jq '.losses[]'

# 暂停（阻塞至训练停止且检查点落盘才返回）
curl -X POST localhost:8000/train/pause

# 续训
curl -X POST localhost:8000/train/resume

# 下载参数 / 完整检查点
curl -OJ localhost:8000/train/parameters     # model_parameters.pt
curl -OJ localhost:8000/train/checkpoint     # checkpoint.pt
```

### 演示：暂停保存失败重试与并发交接

```bash
# 1) 用慢保存配置启动（checkpoint_sleep_seconds=1 放大保存窗口，便于观察）
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"force": true, "config": {"epochs": 2, "checkpoint_every": 10,
          "checkpoint_sleep_seconds": 1.0}}'

# 2) 模拟存储故障：在 runs 目录创建故障注入文件（compose 下为 trainer-runs 卷）
docker compose exec trainer touch /app/runs/INJECT_SAVE_FAILURES
curl -X POST localhost:8000/train/pause
#  -> 500：训练已停，但检查点未能持久化；旧检查点完好，状态为 paused+dirty
curl -s localhost:8000/train/status | jq '.state, .checkpoint_dirty, .last_error'

# 3) 修复存储后重试暂停（保存需 ~1s）；同时在另一个终端立即发续训：
docker compose exec trainer rm /app/runs/INJECT_SAVE_FAILURES
curl -X POST localhost:8000/train/pause &        # 重试暂停（保存者）
curl -X POST localhost:8000/train/resume         # 并发续训：等待保存结束后才放行
#  -> 暂停返回 200 且 state=paused（本作业的结论，不是 running）；
#     续训在保存结束后返回 200 并继续训练。

# 4) 并发暂停共享同一结论：再次注入故障后同时发两个暂停
docker compose exec trainer touch /app/runs/INJECT_SAVE_FAILURES
curl -X POST localhost:8000/train/pause & curl -X POST localhost:8000/train/pause & wait
#  -> 两个请求都得到 500（同一次保存失败的结论），旧检查点完好
docker compose exec trainer rm /app/runs/INJECT_SAVE_FAILURES
curl -X POST localhost:8000/train/pause & curl -X POST localhost:8000/train/pause & wait
#  -> 两个请求都得到 200 + state=paused（共享同一次成功保存）

# 5) 暂停重试保存进行中 force 新开：新开会等待旧保存结束，
#    旧作业的迟到保存不会改写新作业的检查点
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"force": true, "config": {"seed": 99}}'
```

以上交互均可在 `/docs`（Swagger UI）中直接操作；故障注入文件路径在 compose
中为容器内 `/app/runs/INJECT_SAVE_FAILURES`（本地运行为 `$RUNS_DIR/` 下）。

状态机：`idle → running ⇄ pausing → paused → running … → finished`（异常为
`failed`）。`paused` 表示训练已停、可续训；`checkpoint_dirty=true` 表示内存进度
领先于磁盘（上次保存失败），此时重启进程会回退到磁盘上的旧检查点。

### 并发与交接语义（暂停重试 / 作业切换）

- **保存进行中不交接**：`saving=true`（有检查点保存在飞，例如暂停保存失败后的
  重试保存）时，`/train/resume` 与 `/train/start`（含 `force=true`）会**有界等
  待**保存结束再执行；等待超过 `pause_timeout_seconds` 则明确返回 409，可稍后
  重试。绝不会在保存进行中修改或替换正在保存的训练现场。
- **并发暂停共享同一保存结论**：多个 `/train/pause` 同时到达时，第一个请求成
  为"保存者"，其余等待**同一次保存**的结果——要么全部 200（已持久化），要么
  全部 500（保存失败、旧检查点完好、可重试）。暂停响应返回的是**它暂停的那个
  作业**的 `paused` 状态；即使与此同时有续训/新开请求排在后面，也不会返回后来
  作业的状态（如 `running` 或新作业的 `run_id`）。
- **新作业获准启动后，旧保存不越界**：`force=true` 只有在旧作业没有任何保存
  在飞时才获准执行；旧作业的迟到保存不会改写新作业的检查点文件，也不会污染新
  作业的 `checkpoint_dirty`/`last_error` 健康状态（保存完成时会校验所属
  `run_id`，已替换则丢弃该迟到保存）。被丢弃作业的检查点文件会被删除，重启后
  不会恢复已丢弃的作业。
- **保存失败可重试**：暂停保存失败返回 500，训练已停、内存现场完整、磁盘上的
  旧检查点不动；修复存储后再次 `/train/pause` 即可重试持久化，或直接
  `/train/resume` 从内存现场继续训练（下次保存会带上最新进度）。

## 可复现性设计（为什么恢复后逐位一致）

1. **单线程 + 确定性算子**：`torch.set_num_threads(1)` 并开启
   `torch.use_deterministic_algorithms(True)`，同一构建下浮点结果逐位确定。
2. **单一 RNG 流**：新运行先 `torch.manual_seed(seed)`，模型初始化与 Dropout
   都从该流取数；检查点保存 `torch.get_rng_state()`，恢复时原样写回。
3. **每轮打乱可重推**：第 e 轮的样本排列是 `(seed, e)` 的纯函数
   （`random.Random(f"{seed}:{epoch}")`），重启后无需存储即可重建当轮顺序。
4. **批边界检查点**：每个检查点记录 `epoch`、`next_batch`、模型与 Adam 状态、
   RNG 状态、完整损失历史、解析后的配置与 `completed` 标志——即"准备执行第 k
   批"的完整现场。恢复 = 还原现场后继续同一确定性序列。
5. **原子写**：先写 `.checkpoint.pt.tmp.<uuid>` → fsync → `os.replace` 原子改名
   → 目录 fsync。保存中被强退，磁盘上只会是**完整的旧点或完整的新点**；
   启动时清理残留的临时文件。保存失败不触碰旧文件，明确报错，可重试。

> 注意：逐位一致的前提是**同环境**（同一 torch 构建/CPU 指令集、单线程）。跨
> 不同 CPU 架构或 torch 版本不保证逐位一致，但同机同镜像（如本 compose）成立。

## 不中断对照与退出重启示例

### 自动化端到端验证（推荐）

```bash
python3 scripts/verify_repro.py
```

脚本会真实拉起 uvicorn 服务进程，依次验证（详见脚本 docstring）：
A 不中断基线；B 暂停+续训；C 强退+重启+续训；D 并发续训只放行一个；
E1 保存失败→报错→旧点保留→重试成功；E2 保存失败后强退→回退旧点恢复；
F 慢保存不挡查询；G 保存进行中强退→恢复完整旧/新点；H 基础接口与 /docs；
I 暂停重试的保存进行中发续训→续训等待保存结束、暂停返回本作业的 paused；
J 暂停重试的保存进行中 force 新开→新开等待、旧保存不越界、新作业正常复现；
K 并发暂停共享同一保存结论（失败同 500、重试同 200）。
B–E、G、I–K 的逐批损失与最终参数均与各自不中断基线**逐位一致**才判通过。

> 本环境的 seccomp 策略禁止向其他进程发信号，脚本用 `ENABLE_TEST_CRASH_ENDPOINT=1`
> 开启的测试专用 `POST /test/crash`（`os._exit`，立即死亡、无任何清理）模拟强退；
> 恢复语义与 SIGKILL 等价。该端点默认关闭，切勿在生产开启。

### Docker 手动退出重启示例

```bash
docker compose up -d --build
curl -X POST localhost:8000/train/start -H 'Content-Type: application/json' \
     -d '{"config": {"epochs": 8}}'

docker kill resumable-trainer          # 模拟抢占/强退（真实 SIGKILL）
docker compose start                   # 重启容器（检查点在命名卷中）
curl -s localhost:8000/train/status    # state=paused, recovered_after_restart=true
curl -X POST localhost:8000/train/resume
curl -s localhost:8000/train/status    # 直至 state=finished
```

另开一个终端可对同一数据集做一次不中断对照（`force=true` 重跑同配置），用
`GET /train/losses` 与 `GET /train/parameters` 对比两边逐批损失与参数完全一致。

## 实测结果

（以下为 `python3 scripts/verify_repro.py` 在本仓库环境下的真实输出摘要）

```
    workdir: /tmp/verify-repro-kabve2pq
    
    == H: service basics (idle server) ==
      [PASS] H1 GET /
      [PASS] H2 GET /docs (Swagger UI)
      [PASS] H3 parameters before any checkpoint -> 404
      [PASS] H4 resume with no run -> 404
      [PASS] H5 pause while idle -> 409
    
    == A: uninterrupted baseline ==
      [PASS] A baseline finished - 95 batches, final loss 0.027771
    
    == B: pause + resume ==
      [PASS] B1 pause returns only after checkpoint persisted - pause latency 0.03s, stopped at batch 14
      [PASS] B2 resume accepted
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] B3 pause+resume == uninterrupted
    
    == D: concurrent resume starts exactly one trainer ==
      [PASS] D1 concurrent resumes -> one 200, one 409 - got [200, 409]
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] D2 resumed run == uninterrupted
    
    == E1: checkpoint save failure keeps old checkpoint, retry works ==
      [PASS] E1 pause with failing save -> 500 - training is stopped but the checkpoint could not be persisted; the previous checkpoint is intact on 
      [PASS] E1 status shows dirty checkpoint + error
      [PASS] E1 old checkpoint untouched
      [PASS] E1 retried pause -> 200, checkpoint persisted
      [PASS] E1 resume after retry
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E1 run after save-failure == uninterrupted
    
    == F: slow checkpoint saves do not block queries ==
      [PASS] F1 queries respond fast while a 1s save is in flight - worst status latency 2ms
      [PASS] F2 pause during slow saves
      [PASS] F3 slow-save run finishes
    
    == A2: handoff-scenarios baseline (CFG_HANDOFF) ==
      [PASS] A2 handoff baseline finished - 38 batches
    
    == I: pause save-retry hands off to a concurrent resume ==
      [PASS] I1 pause with failing save -> 500 - training is stopped but the checkpoint could not be persisted; the previous chec
      [PASS] I2 resume during save-retry waits for the save, then resumes - got 200 after 1.02s
      [PASS] I3 retried pause returns its own run's paused state (not running) - got 200 paused
        losses: 38 entries, exact match
        params: all tensors bit-identical (sha256 2a6e15fb8366863d == 2a6e15fb8366863d)
      [PASS] I4 run after pause/retry/resume == baseline
    
    == J: pause save-retry hands off to a concurrent force-start ==
      (baseline: seed=99 config, uninterrupted)
      [PASS] J1 pause with failing save -> 500
      [PASS] J2 force-start during save-retry waits, then starts a new run - got 200
      [PASS] J3 retried pause returns its own run's paused state - got 200 paused run=73f0cd1d24f7
      [PASS] J4 discarded run's checkpoint is gone once the new run starts - got 404
      [PASS] J5 late save of old run did not overwrite the new checkpoint - checkpoint run_id=1b02148fe15f
        losses: 38 entries, exact match
        params: all tensors bit-identical (sha256 f6e65b10803c6346 == f6e65b10803c6346)
      [PASS] J6 new run == seed=99 baseline
    
    == K: concurrent pauses share one save conclusion ==
      [PASS] K1 concurrent pauses with failing save -> both get the failure - got [500, 500]
      [PASS] K2 checkpoint still dirty, run paused
      [PASS] K3 concurrent pause retries -> both get the success (paused) - got [200, 200] ['paused', 'paused']
      [PASS] K4 checkpoint persisted by the shared save
      [PASS] K5 resume after concurrent pauses
        losses: 38 entries, exact match
        params: all tensors bit-identical (sha256 2a6e15fb8366863d == 2a6e15fb8366863d)
      [PASS] K6 run after concurrent pauses == baseline
    
    == C: hard crash + restart recovers from latest checkpoint ==
      [PASS] C1 server crashed mid-run
      [PASS] C2 restart recovers paused state from checkpoint - recovered at batch 10/95
      [PASS] C3 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] C4 kill+restart == uninterrupted
    
    == E2: save failure, then hard crash -> old checkpoint used ==
      [PASS] E2 pause with failing save -> 500
      [PASS] E2 restart falls back to old complete checkpoint - recovered at batch 6/95
      [PASS] E2 resume after restart
        losses: 95 entries, exact match
        params: all tensors bit-identical (sha256 2397ddd7b6b8b70b == 2397ddd7b6b8b70b)
      [PASS] E2 save-fail+kill == uninterrupted
    
    == G: hard crash in the middle of a checkpoint save ==
      (baseline: identical config, uninterrupted)
      [PASS] G1 server crashed during save
      [PASS] G2 no torn checkpoint / stale temp cleaned - leftovers: none
      [PASS] G3 restart recovers a complete checkpoint (old or new) - recovered at batch 1/19
      [PASS] G3b resume after restart
        losses: 19 entries, exact match
        params: all tensors bit-identical (sha256 1353a7415c7cae90 == 1353a7415c7cae90)
      [PASS] G4 kill-mid-save == uninterrupted
    
    ================ SUMMARY ================
      50/50 checks passed（逐项明细同上，全部 PASS）

```

另做了 30 轮随机交错压力自测（并发 pause/resume/force-start/查询 + 随机故障
注入）：无死锁、无异常响应、损失历史连续完整，最终可正常暂停持久化并跑完。

## 限制与未完成事项

- 单作业、单进程、单 worker：多副本/多 worker 部署会破坏单训练者语义（需引入
  外部锁与共享存储，未实现）。
- 逐位一致仅保证同环境同构建（见上节）；未做跨机器/跨 torch 版本对齐。
- 检查点只保留最新一份（原子替换保证新旧二选一）；未做历史版本保留/滚动清理。
- `force=true` 启动新运行会丢弃旧检查点（先等旧作业的保存在飞结束，再删文件）：
  若新运行在写出第一个检查点前崩溃，则无点可恢复（状态回到 `idle`），只能重新
  开始——这是刻意语义，避免"已丢弃的旧运行"在重启后被悄悄恢复。
- `GET /train/parameters` 取自最近**已持久化**检查点，运行中可能落后最多
  `checkpoint_every` 批；未提供"当前内存参数"下载（避免与训练线程竞争）。
- 本环境无 docker，镜像构建与 compose 未实际执行；Dockerfile/compose 与本地实
  测使用完全相同的启动命令与依赖版本，但未在真实 Docker 中跑过验证脚本。
- 暂停等待上限为 `pause_timeout_seconds`（默认 30s），超时返回 504，训练仍会
  在批边界停下，可轮询状态确认。
- `resume`/`start` 等待在飞保存的上限同样取 `pause_timeout_seconds`；保存异常
  缓慢时返回 409（明确拒绝），客户端可稍后重试——未做无限等待或排队取消。
