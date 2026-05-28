#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# 设置路径
IMAGE_DIR="/home/gpu1-user13/dns/ldm_ptqd_brecq/20250527_030025/test_image"
REF_BATCH="/home/gpu1-user13/imagenet/VIRTUAL_imagenet256_labeled.npz"
# TEA_PSNR_DIR="./teacher_images"

# 设置输出NPZ文件路径
SAMPLE_BATCH="/home/gpu1-user13/dns/ldm_ptqd_brecq/20250527_030025/test_image/sample_batch.npz"
# SAMPLE_BATCH="/home/gpu1-user13/ncq/ldm_ptqd_brecq_w4a8/20250402_185728/test_image/img_cls_compressed.npz"
# REF_BATCH="./ref_batch.npz"
# TEA_PSNR_BATCH="/home/gpu1-user13/ncq/ldm_ptqd_brecq_w4a8/teacher_ddim20_test_image/sample_batch.npz"
TIMESTAMP="20250527_030025"
# 设置评估结果输出目录
# OUTPUT_DIR="./results"
# mkdir -p $OUTPUT_DIR

# 转换样本图片为NPZ
echo "正在将样本图片转换为NPZ格式..."
python create_npz.py $IMAGE_DIR $SAMPLE_BATCH
if [ $? -ne 0 ]; then
    echo "样本图片转换失败，脚本终止"
    exit 1
fi

# # 转换参考图片为NPZ（如果需要）
# echo "正在将参考图片转换为NPZ格式..."
# python create_npz.py $REF_DIR $REF_BATCH
# if [ $? -ne 0 ]; then
#     echo "参考图片转换失败，脚本终止"
#     exit 1
# fi

# # 转换教师图片为NPZ（如果需要）
# echo "正在将教师图片转换为NPZ格式..."
# python create_npz.py $TEA_PSNR_DIR $TEA_PSNR_BATCH
# if [ $? -ne 0 ]; then
#     echo "教师图片转换失败，脚本终止"
#     exit 1
# fi

# 运行评估程序
echo "开始运行评估程序..."
python evaluator.py $REF_BATCH $SAMPLE_BATCH --timestamp $TIMESTAMP
# python ncqtest/evaluator.py $REF_BATCH $SAMPLE_BATCH --tea_psnr_batch $TEA_PSNR_BATCH --timestamp $TIMESTAMP 
# python ncqtest/evaluator.py $REF_BATCH $SAMPLE_BATCH --output_dir $OUTPUT_DIR

echo "评估完成，结果保存在 $OUTPUT_DIR"