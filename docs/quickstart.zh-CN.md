# 中文快速开始

LocoManiLab 把人形机器人研究中的数据准备、模型适配和本地进程启动连接起来。模型、仿真器、机器人控制器仍各自使用独立环境。先运行下面的 CPU 示例；此步骤不需要子模块、GPU、机器人或模型权重。

## 安装并跑通数据闭环

需要 Python 3.10 或更新版本，请使用独立的 Conda（或 Miniforge）或 uv 环境，不要使用系统自带的 Python。在源码仓库执行：

```bash
conda create -y -n locomanilab python=3.10
conda activate locomanilab
python -m pip install -e '.[data]'
lab init demo-workspace
cd demo-workspace
lab sample-data --out datasets/example
lab validate datasets/example
lab prepare datasets/example --merge-only --out datasets/prepared
lab validate datasets/prepared
```

样例是两段很短的合成序列及小尺寸视频，用于确认安装、数据读取和准备流程；不能用来评价机器人能力或代表正式训练数据。输出保存在当前工作区。

当前 `prepare` 只接受各输入一致的正整数 FPS。29.97 FPS 等合法原始数据仍可保留并用 `validate` 检查，但 `prepare` 会在写入前拒绝，不会截断帧率或自动重采样。

使用 uv 时，把前三条命令换成 `uv venv --python 3.10 .venv`、`source .venv/bin/activate` 和 `uv pip install -e '.[data]'`。

## 下载发布的数据、资产和模型

基准数据、仿真资产清单、GR00T/DM0.5/XR-1 发布模型和 G1 控制器权重托管在 Hugging Face 数据集仓库 [`GammaRobotics/LocoManiLab`](https://huggingface.co/datasets/GammaRobotics/LocoManiLab)，按需下载：

```bash
python -m pip install -e '.[data,hf]'
python scripts/hf/download.py list
python scripts/hf/download.py dataset --task IThorKitchenOpenFridge_scene13 --out datasets/hf
python scripts/hf/minimal_test.py --work hf-minimal-test     # 最小下载与测试单元
```

仿真资产用 `scripts/hf/download.py assets` 校验 sha256 后解压到已初始化的仿真仓库；模型用 `checkpoint gr00t|dm05|xr1` 下载。目录结构、资产预设、各模型的使用要求和测试阶段见[数据、资产与模型](hf_resources.md)。

## 再选择模型和运行时

在源码仓库执行 `python3 scripts/setup/profiles.py --list` 查看依赖分组。`python3 scripts/setup/profiles.py act` 只显示计划；追加 `--apply` 才初始化该组缺失的子模块。版本来自主仓的 gitlink，不会自动追最新分支。部分 fork 可能仍需要访问权限。

在工作区的 `env.local.yaml` 填入本机实际使用的 checkout 和本地权重路径，再先用 `--dry-run` 检查模型命令。新生成的配置使用 schema v2：相对路径以配置目录为基准；已有旧配置保留原来的路径规则。配置和依赖环境不随模型权重共享。

可先执行 `lab doctor --model act --stage train` 查看所需配置。ACT/DP 还需要配置 `paths.lerobot_data_root`、`paths.lerobot_outputs` 并独立安装 LeRobot 的 uv 环境；正式训练前必须把符合模型动作约定的机器人数据通过 `convert` 转换，不能直接使用上述合成样例。详见[本地训练](training.md)。

仿真和真机采集统一使用 `g1_backend` 的控制器与记录层，仿真环境由 `humanoid_benchmark` 提供。`--backend heft` 仍保留；移除的是独立 HeFT 研究源码依赖，不影响控制器选项或兼容的历史数据准备。已有 GR00T/FastWAM 完整真机部署，项目推荐沿用原生 GR00T WholeBodyControl 的部署最佳实践，与统一采集入口分开。

当前发布验证范围以 CPU 数据流程为主；模型、仿真和真机集成尚未在本次整理中逐一做完整验证。在线 RL 的旧入口缺少对应实现，不能当作可用能力。详见[模型矩阵](models.md)、[依赖说明](../dependencies/README.md)和[部署边界](deployment.md)。

主快捷命令为 `lab`，也可用 `python -m locomanilab` 调用。配置使用 `LOCOMANILAB_CONFIG`、`LOCOMANILAB_WORKSPACE`；仿真默认路径为 `external/humanoid_benchmark`。已有配置和数据的兼容规则见[迁移说明](migration.md)。
