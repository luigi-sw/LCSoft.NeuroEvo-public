# LCSoft NeuroEvo - FPGA-based Neural Network Accelerator

## Overview

Complete FPGA-based AI accelerator implementation for neural network inference and training. This project implements a **custom hardware accelerator** from RTL to PyTorch integration, including:

- **Custom RTL design** (Verilog) - Systolic array, DMA, memory controllers
- **PCIe & DDR controllers** - Full implementation without vendor IPs (PHY only)
- **Windows/Linux kernel driver** - Complete with scheduler and HAL
- **PyTorch integration** - Direct `torch.compile()` backend
- **Multi-FPGA support** - Distributed execution across multiple devices
- **Model loaders** - SafeTensors and GGUF with quantization

**Target Performance**: 100+ GFLOPS @ 200MHz (competitive with entry-level GPUs)

---

## Quick Start

### Prerequisites
```bash
# Linux kernel headers
sudo apt-get install build-essential linux-headers-$(uname -r)

# Python & PyTorch
pip install torch pybind11
```

### Build & Install (3 steps)
```bash
# 1. Build HAL library
cd driver && gcc -o libhal.so hal.c -shared -fPIC
sudo cp libhal.so /usr/local/lib/ && sudo ldconfig

# 2. Build & load kernel driver
make && sudo insmod ai_driver.ko
sudo chmod 666 /dev/neuroevo

# 3. Install PyTorch extension
cd ../pytorch && python setup.py install
```

### Run Your First Model
```python
import torch
from pytorch.accelerator import AcceleratorLinear, init

# Initialize hardware
init()

# Replace nn.Linear with AcceleratorLinear
layer = AcceleratorLinear(1024, 1024)

# Run inference
input = torch.randn(32, 1024)
output = layer(input)  # Executes on FPGA!
```

### Or use torch.compile()
```python
from pytorch import dynamo_backend
dynamo_backend.register()

model = torch.nn.Sequential(
    torch.nn.Linear(1024, 1024),
    torch.nn.GELU()
)

# Compile for FPGA
opt_model = torch.compile(model, backend='NeuroEvo')
output = opt_model(input)  # Auto-offload to FPGA
```

---

## Features

### Hardware (RTL)
- **32x32 Systolic Array** (dense) / **64x64 Sparse** (flagship)
- **FP8 E4M3** / BF16 / INT16 precision
- **2:4 Structured Sparsity** (50% compute savings on flagship)
- **Hardware Operator Fusion** (Linear+Bias+GELU)
- **2D DMA Engine** with strided access
- **Multi-channel DDR** (4-8 channels, up to 200 GB/s bandwidth)
- **Custom PCIe controller** (Gen3 x8)
- **Multi-FPGA interconnect** (100G Ethernet/Aurora)

### Software Stack
- **Windows/Linux Kernel Driver** with job scheduler
- **HAL (Hardware Abstraction Layer)** in C
- **PyTorch Backend** for `torch.compile()`
- **MLIR Compiler** with auto-tuning
- **Distributed Runtime** for multi-FPGA
- **Model Loaders**: SafeTensors, GGUF (with Q4_0/Q8_0 → FP8 transcoding)

### Optimization Features
- **Graph Fusion** (Linear → Add → GELU patterns)
- **Auto-tuning** (tile size optimization with real hardware benchmarking)
- **Memory Striping** (RAID-0 style across DDR channels)
- **Zero-copy** data transfer (mmap BAR2)

---

## Architecture Overview

- [Architecture](src/docs/PROJECT_ARCHITECTURE.md)

---

## Documentation

### Getting Started
- [Build Instructions](BUILD_INSTRUCTIONS.md) - Step-by-step build guide
- [Project Overview](src/docs/PROJECT_OVERVIEW.md) - High-level project description
- [Quick Tutorial](src/docs/TUTORIAL.md) - 10-minute walkthrough

### Architecture
- [Hardware Architecture](src/docs/HARDWARE_ARCHITECTURE.md) - RTL design deep-dive
- [Software Architecture](src/docs/SOFTWARE_ARCHITECTURE.md) - Driver & stack design
- [Memory System](src/docs/MEMORY_SYSTEM.md) - DDR controllers & NUMA

### Component Details
- [RTL Design Guide](src/docs/RTL_DESIGN.md) - Verilog modules explained
- [Driver Guide](src/docs/DRIVER_GUIDE.md) - Kernel driver internals
- [HAL API Reference](src/docs/HAL_API.md) - C API documentation
- [PyTorch Integration](src/docs/PYTORCH_INTEGRATION.md) - Backend implementation
- [MLIR Compiler](src/docs/MLIR_COMPILER.md) - Optimization passes

### Performance & Tuning
- [Performance Analysis](src/docs/PERFORMANCE.md) - Benchmarks & bottlenecks
- [Auto-Tuning Guide](src/docs/AUTO_TUNING.md) - Tile size optimization
- [Distributed Execution](src/docs/DISTRIBUTED.md) - Multi-FPGA setup

---

## Project Structure

```
rtl/
  core/            # AI Core (systolic array, PE, vector proc)
  control/         # Control logic (decoder, DMA, FIFOs)
  memory/          # Memory subsystem (DDR ctrl, interconnect)
  interfaces/      # PCIe, chip-link
  top/             # Top-level integration

driver/
  ai_driver.c          # Kernel module (scheduler, IRQ)
  hal.c                # Hardware Abstraction Layer
  ai_driver_windows.c  # Windows WDM driver

pytorch/
  accelerator_ops.cpp  # C++ extension (uses HAL)
  dynamo_backend.py    # torch.compile() backend
  accelerator.py       # AcceleratorLinear layer

tools/
  compiler.py          # Instruction compiler
  mlir_compiler.py     # MLIR-style optimizer

loaders/
  safetensors_loader.cpp
  gguf_loader.cpp      # Quantization transcoding
  py_bindings.cpp      # Python bindings

runtime/
  distributed.py       # Multi-FPGA orchestration

tb/                    # Testbenches (Verilog)

docs/                  # Documentation
```

---

## Performance Targets

### Compute
- **Peak Performance**: 100-200 GFLOPS (INT16) / 400 GFLOPS (INT8)
- **Sustained**: 70-80% of peak (with good tiling)
- **Latency**: < 1ms for 1024x1024 GEMM

### Memory
- **Bandwidth**: 200 GB/s (4x DDR4-2400) to 512 GB/s (8x GDDR6)
- **Capacity**: 16-64 GB (depending on DIMM/chip config)
- **Latency**: ~100ns (on-chip SRAM) to ~10μs (DDR)

### Comparison (Ballpark)
| Device | GFLOPS (INT8) | Memory BW | Power |
|--------|---------------|-----------|-------|
| **This Project (PRO)** | **~200** | **200 GB/s** | **~25W** |
| **This Project (FLAGSHIP)** | **~400** | **512 GB/s** | **~50W** |
| NVIDIA T4 | 260 | 320 GB/s | 70W |
| NVIDIA A100 | 624 | 1555 GB/s | 250W |
| Google TPU v4 | 275 | 1200 GB/s | 180W |

*Note: FP32 performance will be lower (~50 GFLOPS) due to array size constraints*

---

## Development Status

### Completed
- [x] RTL design (systolic array, DMA, decoders)
- [x] PCIe controller (digital logic, no vendor IP)
- [x] DDR controllers (GDDR6, LPDDR5, DDR4/5 ready)
- [x] Linux kernel driver with scheduler
- [x] HAL library
- [x] PyTorch C++ extension
- [x] Dynamo backend for torch.compile()
- [x] MLIR compiler with fusion & auto-tune
- [x] SafeTensors & GGUF loaders
- [x] Distributed runtime
- [x] Complete integration (PyTorch → HAL → Driver → HW)

### In Progress
- [ ] Hardware synthesis (targeting Xilinx VCU118)
- [ ] FPGA bitstream generation
- [ ] PCIe PHY integration (using Xilinx PCIe core)
- [ ] End-to-end hardware validation

### Future Work
- [ ] Training support (backward pass on FPGA)
- [ ] FP8 E5M2 (for gradients)
- [ ] Transformer-specific optimizations
- [ ] Direct HBM support (for Alveo boards)
- [ ] Windows driver completion
- [ ] ONNX Runtime integration

---

## Hardware Requirements

### FPGA Target
- **Minimum**: Xilinx Kintex UltraScale+ (XCKU5P)
  - Resources: ~280K LUTs, ~500K FFs, ~1K BRAMs
  - PCIe: Gen3 x8
  - Memory: 4x DDR4 SODIMM slots
  
- **Recommended**: Xilinx VCU118 (VU9P)
  - Resources: ~1.18M LUTs, ~2.36M FFs, ~4K BRAMs
  - PCIe: Gen3 x16
  - Memory: 8x DDR4 slots + 8GB DDR4 on-board

- **Flagship**: Xilinx Alveo U280
  - HBM2 (32GB @ 460 GB/s)
  - PCIe Gen4 x16
  - Pre-integrated PCIe/DDR PHYs

### Host System
- **OS**: Linux (Ubuntu 20.04+ or similar)
- **CPU**: Any x86_64 with PCIe slot
- **RAM**: 16GB+ recommended
- **PCIe**: Gen3 x8 or better

---

## Contributing

This is a research/educational project. Contributions welcome for:
- Additional optimization passes
- Alternative memory controllers
- Extended PyTorch op support
- Windows driver improvements
- Documentation enhancements

---

## License

CC BY-NC-ND License - See [LICENSE](LICENSE) file

---

## Citation

If you use this project in research, please cite:
```
@misc{ai_accelerator_2025,
  title={Custom FPGA-based AI Accelerator with PyTorch Integration},
  year={2025},
  url={https://github.com/luigi-sw/LCSoft.NeuroEvo-public}
}
```

---

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/luigi-sw/LCSoft.NeuroEvo-public/issues)
- **Discussions**: [GitHub Discussions](https://github.com/luigi-sw/LCSoft.NeuroEvo-public/discussions)
- **Documentation**: [Full Docs](src/docs/)

