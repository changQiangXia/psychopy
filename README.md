# CPT 注意力实验系统（PsychoPy）

本项目实现了一个可直接运行的 CPT（持续注意）实验系统，基于 PsychoPy，已对接补充资料中的真实素材与指标口径，支持：

- 练习模式（1 分钟自动结束）
- 正式测试（初测 / 复测）
- 复测身份校验（姓名 + 编号需与初测一致）
- 管理者中心（表格化查看 + Excel 导出）
- Windows 可执行版打包（PyInstaller）

---

## 1. 项目结构

核心文件：

- `grad_cpt.py`：主程序入口（实验流程 + 结果管理）
- `build_exe.ps1`：Windows 可执行版打包脚本
- `supplementary_info.zip`：补充资料（程序可自动解压）
- `requirements_from_images.md`：初始图片需求整理备查

运行后自动生成：

- `data/raw_trials/`：逐试次原始数据
- `data/results/初测结果.csv`
- `data/results/复测结果.csv`
- `data/results/总结果.csv`
- `data/results/CPT数据导出_*.xlsx`

---

## 2. 运行环境

推荐环境：

- Conda 环境名：`psychopy_pip_env`
- Python：3.10（已在当前项目实践中验证）
- 操作系统：Windows

激活环境：

```powershell
conda activate psychopy_pip_env
```

---

## 3. 素材加载规则

程序优先按以下顺序查找素材：

1. 自动解压并使用 `supplementary_info.zip` 中的素材
2. 若补充包不存在，则回退到项目内 `stimuli/` 目录

支持的阶段目录结构（二选一）：

- 中文结构：
  - `初测/city`
  - `初测/mountain`
  - `后测/city`
  - `后测/mountain`
- 英文结构：
  - `pre/city`
  - `pre/mountain`
  - `post/city`
  - `post/mountain`

支持的图片格式：

- `png/jpg/jpeg/bmp/tif/tiff/webp`

---

## 4. 快速开始（交互模式）

```powershell
conda activate psychopy_pip_env
python .\grad_cpt.py
```

主菜单包含：

1. 练习模式
2. 正式测试（初测 / 复测）
3. 管理者中心
4. 退出

---

## 5. 命令行模式

### 5.1 练习模式

```powershell
python .\grad_cpt.py --mode practice --windowed
```

### 5.2 初测

```powershell
python .\grad_cpt.py --mode initial --name 张三 --participant-id 001 --sex 男 --grade 大一 --windowed
```

### 5.3 复测（需与初测姓名+编号一致）

```powershell
python .\grad_cpt.py --mode retest --name 张三 --participant-id 001 --windowed
```

### 5.4 管理者中心

```powershell
python .\grad_cpt.py --mode manager
```

### 5.5 仅做逻辑检查（不打开实验窗口）

```powershell
python .\grad_cpt.py --mode practice --dry-run
python .\grad_cpt.py --mode initial --name demo --participant-id 1 --sex 男 --grade 大一 --dry-run
```

---

## 6. 关键参数说明

可用参数（`grad_cpt.py`）：

- `--mode`：`menu|practice|initial|retest|manager`
- `--name`：姓名（初测/复测）
- `--participant-id`：编号（初测/复测）
- `--sex`：`男|女`（初测必填）
- `--grade`：年级（初测必填）
- `--materials-root`：手动指定素材根目录
- `--data-dir`：输出目录，默认 `data`
- `--seed`：随机种子
- `--main-trials`：正式测试试次数，默认 `497`
- `--practice-seconds`：练习时长（秒），默认 `60`
- `--trial-duration`：单试次时长，默认 `1.2`
- `--fade-duration`：线性融合时长，默认 `0.8`
- `--go-ratio`：城市刺激比例，默认 `0.9`
- `--response-key`：Go 按键，默认 `space`
- `--windowed`：窗口模式运行
- `--dry-run`：仅构建试次不跑实验
- `--prepare-materials`：仅解压补充资料素材后退出

---

## 7. 实验流程说明

### 7.1 练习模式

- 时长 1 分钟（按 `practice-seconds / trial-duration` 折算试次）
- 与正式流程一致（按键规则一致）
- 自动记录原始试次数据

### 7.2 初测

- 录入字段：姓名、性别、年级、编号
- 使用“初测”素材进行正式任务
- 结束后写入：
  - 原始数据：`data/raw_trials/*_initial_*.csv`
  - 结果数据：`data/results/初测结果.csv`
  - 同步刷新：`data/results/总结果.csv`

### 7.3 复测

- 录入字段：姓名、编号
- 必须与初测已有记录完全匹配（姓名 + 编号）
- 使用“后测”素材进行正式任务
- 结束后写入：
  - 原始数据：`data/raw_trials/*_retest_*.csv`
  - 结果数据：`data/results/复测结果.csv`
  - 同步刷新：`data/results/总结果.csv`

---

## 8. 管理者中心

管理者中心功能：

- 查看初测结果（表格窗口）
- 查看复测结果（表格窗口）
- 查看总结果（表格窗口）
- 下载数据（Excel）

说明：

- 表格窗口支持横向/纵向滚动，适合列较多场景
- 若运行环境缺少 tkinter，程序会自动回退为“控制台预览 + 弹窗提示”

---

## 9. 数据文件字段说明

### 9.1 原始试次数据（`data/raw_trials/*.csv`）

典型列：

- `trial_index`
- `condition`（city/mountain）
- `stimulus_file`
- `response_key`
- `responded`（0/1）
- `rt_s`
- `correct`（0/1）
- `error_type`（commission/omission/空）
- `trial_duration_s`
- `phase`（practice/initial/retest）
- `name`
- `participant_id`
- `sex`
- `grade`
- `start_time`

### 9.2 初测/复测结果（`data/results/初测结果.csv` / `复测结果.csv`）

典型列：

- `测试时间`
- `阶段`
- `编号`
- `姓名`
- `性别`
- `年级`
- `试次数`
- `正确率`
- `虚报率CER`
- `漏报率OER`
- `平均反应时RT`
- `反应时变异RTCV`
- `辨别力d'`
- `完成状态`
- `原始数据文件`

### 9.3 总结果（`data/results/总结果.csv`）

以“编号 + 姓名”为主键，按最近记录聚合：

- 基础信息：`编号, 姓名, 性别, 年级`
- 初测列：`初测时间, 初测CER, 初测OER, 初测RT, 初测RTCV, 初测d'`
- 复测列：`复测时间, 复测CER, 复测OER, 复测RT, 复测RTCV, 复测d'`

---

## 10. 指标口径（严格按补充资料 formula.png）

- `CER`（虚报率）  
  错误按键的山脉试次 / 总山脉试次

- `OER`（漏报率）  
  未按键的城市试次 / 总城市试次

- `RT`（平均反应时）  
  所有正确城市试次 RT 的均值

- `RTCV`（反应时变异）  
  所有正确城市试次 RT 的标准差

- `d'`（辨别力）  
  `d' = Z(击中率) - Z(虚报率)`  
  程序对 0/1 概率做了标准校正（避免无穷值）

---

## 11. 打包可执行版（Windows）

推荐使用 `onedir`，稳定性高于 `onefile`（PsychoPy 依赖较重）。

执行：

```powershell
conda activate psychopy_pip_env
powershell -ExecutionPolicy Bypass -File .\build_exe.ps1 -CleanFirst
```

产物：

- `dist\CPTManager\CPTManager.exe`

脚本行为：

- 自动检测并安装 `pyinstaller`（若未安装）
- 自动拷贝 `supplementary_info.zip` 到发布目录

---

## 12. 常见问题

### Q1：复测提示“姓名和编号与初测不一致”

排查顺序：

1. 确认已先完成初测
2. 确认 `data/results/初测结果.csv` 存在
3. 复测输入的姓名与编号需与初测完全一致

### Q2：只看到初测结果，没有复测结果

说明复测未成功完成或被校验拦截。  
检查 `data/results/复测结果.csv` 是否生成。

### Q3：运行时报素材目录错误

确保项目根目录有 `supplementary_info.zip`，或手动指定：

```powershell
python .\grad_cpt.py --materials-root "你的素材根目录"
```

### Q4：控制台出现中文乱码

这是终端编码问题，不影响 CSV 与实验逻辑。  
可优先使用 GUI 菜单与结果窗口查看。

---

## 13. 建议验收清单

建议每次交付前最少走一遍：

1. 练习模式完整跑通
2. 初测完整跑通并写入结果
3. 复测完整跑通并写入结果
4. 总结果正确合并初测与复测
5. 管理者中心表格可打开三类结果
6. Excel 导出成功且包含 4 个工作表（初测/复测/总结果/原始试次）

---

## 14. 版权与用途说明

本仓库用于科研/教学流程实现示例。  
实验素材版权归原素材来源方所有，请按你的研究合规要求使用。

