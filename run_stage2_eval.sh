#!/bin/bash
# Stage 2 Evaluation: Inference + Visualization + Benchmark
set -e

CONDA_ENV=/home/juyeon/miniconda3/envs/chartinternvl
WORK_DIR=/NetDisk/juyeon/DocSP
VIZ_DIR=/NetDisk/juyeon/vlm_viz/redundancy_viz
STAGE2_MODEL=${WORK_DIR}/outputs/stage2_multinode

export PATH=${CONDA_ENV}/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

cd ${WORK_DIR}

echo "============================================"
echo "=== 1. Stage 2 Inference Test ==="
echo "============================================"
${CONDA_ENV}/bin/python test_inference_stage2.py

echo ""
echo "============================================"
echo "=== 2. Stage 2 Visualization ==="
echo "============================================"
cd ${VIZ_DIR}
# Update config to Stage 2 model
${CONDA_ENV}/bin/python -c "
import re
with open('config.py') as f:
    content = f.read()
content = re.sub(
    r'INTERNVL_MODEL_PATH = .*',
    'INTERNVL_MODEL_PATH = \"/NetDisk/juyeon/DocSP/outputs/stage2_multinode\"',
    content
)
with open('config.py', 'w') as f:
    f.write(content)
print('config.py updated to Stage 2 model')
"
${CONDA_ENV}/bin/python visualize_internvl_chartqa.py --output_dir outputs/stage2_multinode_viz --num_samples 5

echo ""
echo "============================================"
echo "=== 3. ChartQA Benchmark ==="
echo "============================================"
cd ${WORK_DIR}
${CONDA_ENV}/bin/python benchmark_chartqa.py --model_path ${STAGE2_MODEL}

echo ""
echo "============================================"
echo "=== All evaluations complete! ==="
echo "============================================"
