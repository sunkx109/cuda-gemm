# Architecture & 开发说明

本文档说明 `cuda-gemm` 插件的结构、构建流程、绑定机制，以及如何添加新的 GEMM 变体。

## 目录结构

```
cuda-gemm/
├── csrc/                      # 所有 C++/CUDA 源码
│   ├── ops.h                  # torch 侧接口声明（gemm_naive/gemm_tiled wrapper 原型）
│   ├── launchers.h            # torch-free 裸指针 launcher 原型 + gemm_launcher_fn 签名
│   ├── ops.cpp                # torchbind 注册 + torch wrapper + 共享 gemm_impl（g++ 编译）
│   └── kernels/               # 各 GEMM 实现（一个文件 = 一个变体，nvcc 编译，不含 torch）
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

调用链分三层，**torch 只出现在最上层**，`.cu` 文件完全不碰 torch：

```
torch.ops.cuda_gemm.gemm_naive          # Python 侧（dispatcher）
└─ gemm_naive        (ops.cpp)          # torch wrapper：一行 dispatch
   └─ gemm_impl      (ops.cpp)          # 共享 helper：校验/contiguous/empty/取 stream
      └─ launch_matmul_naive (kernels/gemm_naive.cu)  # 裸指针 + <<<>>> launch
```

torchbind 注册与 wrapper 都在 `csrc/ops.cpp`：

```cpp
TORCH_LIBRARY(cuda_gemm, m) {            // 声明算子 schema（注册到 dispatcher）
  m.def("gemm_naive(Tensor A, Tensor B) -> Tensor");
  m.def("gemm_tiled(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_gemm, CUDA, m) { // 绑定 CUDA wrapper（定义在同文件 ops.cpp）
  m.impl("gemm_naive", &gemm_naive);
  m.impl("gemm_tiled", &gemm_tiled);
}
```

每个 `gemm_*` wrapper 只是把对应 backend 的 launcher 传给共享的 `gemm_impl`，校验与 tensor 装配逻辑只此一份：

```cpp
torch::Tensor gemm_naive(torch::Tensor A, torch::Tensor B) {
    return gemm_impl(std::move(A), std::move(B), launch_matmul_naive);
}
```

底部保留一个空的 `PYBIND11_MODULE`，使 `import cuda_gemm` 能加载 `.so`，从而触发上面的静态注册——之后 `torch.ops.cuda_gemm.*` 才可用。所以**使用前必须 `import cuda_gemm`**（在 `import torch` 之后）。

当前仅注册前向、CUDA dispatch key，**不含 autograd**。

## 为什么 `.cu` 不 include torch（构建性能）

这是一条刻意的架构约束，**新增 kernel 时务必遵守**：`.cu` 文件只 `#include <cuda_runtime.h>` 和 `launchers.h`，绝不 include `ops.h` / `<torch/extension.h>` / `<ATen/...>`。

原因：nvcc 重新解析 PyTorch 的模板密集头文件极慢——实测单个 `.cu` **~85 s**，且 `-O0` 与 `-O3` 几乎一样，证明时间几乎全花在前端解析而非后端优化。把 torch 留给 `ops.cpp`（g++ 编译，解析这堆头只需几秒）后：

| 场景                  | 之前     | 之后    |
| --------------------- | -------- | ------- |
| 改一个 `.cu` 重编     | ~88 s    | ~3 s    |
| 全量 clean build      | ~90 s    | ~28 s   |

一旦在某个 `.cu` 里 include 了 torch，该文件编译时间立刻回到分钟级，抵消整个优化。launcher 只接 `const float*` + `cudaStream_t`（见 `launchers.h` 的 `gemm_launcher_fn`），正是为了能跨过 torch 的边界。

## 如何添加一个 GEMM 变体

以新增 `gemm_wmma` 为例。kernel 本体写在 torch-free 的 `.cu`，torch 装配复用 `gemm_impl`，无需复制校验代码：

1. **写 launcher**：新建 `csrc/kernels/gemm_wmma.cu`，只 `#include <cuda_runtime.h>` 与 `#include "launchers.h"`（**不要 include torch**）。实现 `__global__ void matmul_wmma_kernel(...)`，以及 host 函数 `void launch_matmul_wmma(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream)`——签名照 `launchers.h` 的 `gemm_launcher_fn`，内部做 `dim3`/grid/`<<<>>>` launch。
2. **声明 launcher**：在 `csrc/launchers.h` 加一行 `void launch_matmul_wmma(...)`。
3. **加一行 dispatch wrapper**：在 `csrc/ops.cpp` 加
   ```cpp
   torch::Tensor gemm_wmma(torch::Tensor A, torch::Tensor B) {
       return gemm_impl(std::move(A), std::move(B), launch_matmul_wmma);
   }
   ```
   并在 `TORCH_LIBRARY` 里 `m.def("gemm_wmma(Tensor A, Tensor B) -> Tensor")`、`TORCH_LIBRARY_IMPL(cuda_gemm, CUDA, m)` 里 `m.impl("gemm_wmma", &gemm_wmma)`。
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
