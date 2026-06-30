# Architecture & 开发说明

本文档说明 `cuda-gemm` 插件的结构、构建流程、绑定机制，以及如何添加新的 GEMM 变体。

## 目录结构

```
cuda-gemm/
├── csrc/                      # 所有 C++/CUDA 源码
│   ├── ops.h                  # 算子接口声明（C++ launcher 原型）
│   ├── ops.cpp                # torchbind 注册：TORCH_LIBRARY + TORCH_LIBRARY_IMPL
│   └── kernels/               # 各 GEMM 实现（一个文件 = 一个变体）
│       ├── gemm_naive.cu
│       └── gemm_tiled.cu
├── tests/                     # 功能单测（正确性）
├── benchmark/                 # 性能脚本（延迟 + TFLOPS）
├── examples/                  # 最小用法示例
├── docs/                      # 文档
├── setup.py                   # 编译安装入口（glob 自动收集 csrc/ 源码）
├── requirements.txt           # 运行/测试依赖
├── requirements-dev.txt       # 开发依赖（pre-commit）
├── pyproject.toml             # ruff 配置（仅工具配置，无 build-system）
├── .clang-format              # C++/CUDA 格式规范
└── .pre-commit-config.yaml    # 提交前检查
```

## 构建流程

`setup.py` 用 `torch.utils.cpp_extension.CUDAExtension` + `BuildExtension`，并 **glob 自动收集** `csrc/` 下所有 `*.cpp` / `*.cu`：

```python
sources = sorted(glob.glob("csrc/**/*.cpp", recursive=True)
                 + glob.glob("csrc/**/*.cu", recursive=True))
```

因此新增一个 kernel 文件后**无需修改 `setup.py`**，重装即可：

```bash
python setup.py develop      # 开发模式，改了源码后重跑
```

> 注意：`include_dirs` 用 `csrc/` 的**绝对路径**——torch 的 ninja 构建在 `build/temp/` 下执行，相对的 `-I` 会解析到错误目录。
> GPU 架构由 `setup.py` 自动探测（`-gencode`），可用环境变量 `TORCH_CUDA_ARCH_LIST` 覆盖。

## torchbind 绑定机制

绑定分两层，都在 `csrc/ops.cpp`：

```cpp
TORCH_LIBRARY(cuda_gemm, m) {            // 声明算子 schema（注册到 dispatcher）
  m.def("gemm_naive(Tensor A, Tensor B) -> Tensor");
  m.def("gemm_tiled(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_gemm, CUDA, m) { // 绑定 CUDA 实现（实现在 csrc/kernels/*.cu）
  m.impl("gemm_naive", &gemm_naive);
  m.impl("gemm_tiled", &gemm_tiled);
}
```

底部保留一个空的 `PYBIND11_MODULE`，使 `import cuda_gemm` 能加载 `.so`，从而触发上面的静态注册——之后 `torch.ops.cuda_gemm.*` 才可用。所以**使用前必须 `import cuda_gemm`**（在 `import torch` 之后）。

当前仅注册前向、CUDA dispatch key，**不含 autograd**。

## 如何添加一个 GEMM 变体

以新增 `gemm_wmma` 为例：

1. **写 kernel**：新建 `csrc/kernels/gemm_wmma.cu`，实现 `__global__` kernel 与 host launcher `torch::Tensor gemm_wmma(torch::Tensor A, torch::Tensor B)`（用 `#include "ops.h"` 复用声明）。
2. **声明接口**：在 `csrc/ops.h` 加 `torch::Tensor gemm_wmma(torch::Tensor A, torch::Tensor B);`。
3. **注册算子**：在 `csrc/ops.cpp` 的 `TORCH_LIBRARY` 里 `m.def("gemm_wmma(Tensor A, Tensor B) -> Tensor")`，并在 `TORCH_LIBRARY_IMPL(cuda_gemm, CUDA, m)` 里 `m.impl("gemm_wmma", &gemm_wmma)`。
4. **重装**：`python setup.py develop`（无需改 `setup.py`，glob 会自动纳入新文件）。
5. **验证**：调用 `torch.ops.cuda_gemm.gemm_wmma(a, b)`；在 `tests/test_gemm.py` 的 `OPS` 列表里加上它即可自动跑全部尺寸的正确性测试。

## 目录职责约定

| 目录       | 职责                                   |
| ---------- | -------------------------------------- |
| `tests/`   | 功能单测（`pytest`，对 `torch.matmul` 比对） |
| `benchmark/` | `bench_gemm.py`（延迟 + TFLOPS）；`profile_kernel.py`（单 kernel，配合 `ncu`） |
| `examples/` | 最小用法示例（不测性能）               |
| `docs/`    | 架构与开发文档                         |

## 性能测试

```bash
python benchmark/bench_gemm.py                          # 默认 512/1024/2048
python benchmark/bench_gemm.py --sizes 512 1024 2048 4096 --iters 50
```

输出每个 kernel 在各尺寸下的延迟（us）与算力（TFLOPS），并附带轻量正确性校验，避免 benchmark 一个写坏的 kernel。

## 单 kernel 分析（Nsight Compute / ncu）

`benchmark/profile_kernel.py` 用 `--kernel` 指定**单个** kernel 独立调用（不跑 `torch.matmul` / 其它变体），便于 `ncu` 抓干净的单 kernel profile。脚本会打印一条可直接复制的 ncu 命令：

```bash
python benchmark/profile_kernel.py --kernel gemm_tiled --size 4096

ncu --set full --target-processes all \
    --kernel-name regex:matmul_tiled_kernel --launch-skip 3 --launch-count 1 \
    -o profile_gemm_tiled \
    python benchmark/profile_kernel.py --kernel gemm_tiled --size 4096
```

- `--kernel-name regex:matmul_<variant>_kernel` 过滤掉 `torch.randn` 等无关 kernel，只采集你的 GEMM kernel。
- `--launch-skip <warmup>` 跳过预热；`--launch-count <repeats>` 限定采集次数（默认 warmup=3、repeats=1）。
- 新增 kernel 变体时，在脚本的 `KERNELS` 与 `DEVICE_KERNEL` 两个映射里加一行即可。
- 若 ncu 报权限错误：`echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`，或用 `sudo` 运行。

## 代码规范（pre-commit）

提交前由 pre-commit 自动检查/格式化（详见 `.pre-commit-config.yaml`）：

- **Python**：ruff（lint + format，配置在 `pyproject.toml`）。
- **C++/CUDA**：clang-format（配置在 `.clang-format`）。
- **通用**：尾随空白、文件末尾换行、行尾统一 LF、YAML/TOML 校验、大文件拦截。

首次接入：

```bash
pip install -r requirements-dev.txt
pre-commit install        # 安装 git hook，此后每次 commit 自动触发
pre-commit run --all-files   # 手动全量检查
```
