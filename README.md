# gc6d_lift3d_traj

将 GraspClutter6D（GC6D）中有效 GT 抓取解码后，在 **相机坐标系（realsense-d415）** 下生成模仿学习轨迹，并导出为可与 Lift3D 对齐的数据格式。大文件默认写入 `/mnt/ssd/ziyaochen/gc6d_lift3d_traj_data`。

## 环境

- **Python**：建议 **3.10–3.12**（`open3d` 对 3.13 可能无 wheel）。
- 依赖：`pip install -r requirements.txt`
- 将 GC6D 数据置于 `configs/default.yaml` 中的 `paths.gc6d_root`，`graspclutter6dAPI` 路径指向本地 clone。

## 1. Lift3D 旋转表示（已核对源码）

- GC6D 离线 **10D action**：`[translation(3), rotation_6d(6), width(1)]`，6D 为 **R 的前两列**（Gram–Schmidt 回 `SO(3)`）。
- 参考：`~/LIFT3D/lift3d/dataset/gc6d_offline_npz.py`、`lift3d/models/grasp_head.py`。
- 本仓库 **状态 `ee_rotations`**：绝对 6D；**动作 `actions_rotation`**：`R_delta = R_{t+1} R_t^T` 再取 6D。

## 2. GC6D 17D 抓取（已核对 graspclutter6dAPI）

| 索引 | 含义 |
|------|------|
| 0 | score |
| 1 | width |
| 2 | height |
| 3 | depth |
| 4:13 | R，行优先 9 个数 |
| 13:16 | translation（中心，相机系） |
| 16 | object_id |

**approach**：`R[:, 0]`（与 GraspNet 视点 x 轴一致）。实现：`gc6d_lift3d_traj/gc6d/grasp_decode.py`。

`loadGrasp` 已在 API 内用摩擦与 **collision 标签**筛除无效抓取；本仓库再按 `score` 排序取 **top_k**。

## 3. 生成数据集

```bash
cd ~/gc6d_lift3d_traj
export PYTHONPATH=.

# 调试：只处理前 K 个样本帧
python scripts/build_gc6d_lift3d_dataset.py --max-samples 5 --output-root /mnt/ssd/ziyaochen/gc6d_lift3d_traj_data

# 全量（极慢）：不传 --max-samples，且 default.yaml 里 dataset.max_samples 为 null
python scripts/build_gc6d_lift3d_dataset.py --output-root /mnt/ssd/ziyaochen/gc6d_lift3d_traj_data

# 断点续跑（已存在的 .npz 跳过）
python scripts/build_gc6d_lift3d_dataset.py --resume ...
```

输出：`episodes/*.npz`、`index/index_train.jsonl`（文件名可由 `dataset.index_filename` 配置）、`metadata/build_run.json`。

**碰撞**：简化平行夹爪三盒 + 桌面高度（点云 z 分位数估计 `table_z_est`）。若 0 episode，可在 `configs/default.yaml` 中增大 `collision.max_points_in_boxes`。

## 4. 自检

```bash
python scripts/sanity_check_episodes.py --index /mnt/ssd/ziyaochen/gc6d_lift3d_traj_data/index/index_train.jsonl
```

## 5. 可视化

```bash
python scripts/visualize_random_episodes.py \
  --index /mnt/ssd/ziyaochen/gc6d_lift3d_traj_data/index/index_train.jsonl \
  --num 10 \
  --out-dir /mnt/ssd/ziyaochen/gc6d_lift3d_traj_data/visualizations
```

## 6. 训练（轨迹策略：模仿 + 目标抓取 + 夹爪）

使用 **点云 mean/max 池化 + EE 状态** 预测每步 **10 维 delta**（3+6+1）及辅助 **10 维 grasp 目标**（与 GT 10D 对齐）。

```bash
python scripts/train_lift3d_gc6d.py \
  --index /mnt/ssd/ziyaochen/gc6d_lift3d_traj_data/index/index_train.jsonl \
  --epochs 20 \
  --out /mnt/ssd/ziyaochen/gc6d_lift3d_traj_data/metadata/traj_policy.pt
```

与 **Lift3D 官方 GC6D 单步 grasp 训练**（`lift3d/tools/train_policy_gc6d.py` + 10D MSE）不同：此处为 **逐步轨迹**；若需完全对齐官方 pipeline，可将单步数据导出为与 `GC6DOfflineNPZDataset` 相同的 `action`(10,) 格式（另写导出脚本即可）。

## 7. 评估

```bash
python scripts/eval_lift3d_gc6d.py \
  --index .../index_train.jsonl \
  --ckpt .../traj_policy.pt \
  --dump-17d-dir .../pred_npy   # 可选：保存拼接 17D 行，供自行对接 API
```

完整 **GC6D AP** 需按 `graspclutter6dAPI/tools/run_benchmark_from_offline_policy.py` 的目录约定为每个 scene 写入 `GraspGroup` `.npy`；本脚本提供指标与可选向量导出。

## 8. 检查脚本

```bash
python scripts/inspect_lift3d_rotation.py
python scripts/inspect_gc6d_grasp_format.py
```

## 9. 测试

```bash
cd ~/gc6d_lift3d_traj && pytest -q
```
