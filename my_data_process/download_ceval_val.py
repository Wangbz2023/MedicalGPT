import os
from datasets import load_dataset

# 1. 环境变量与学术加速（确保 source /etc/network_turbo 已经执行）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 直接指向最终存放 csv 的物理目录
save_dir = "/root/autodl-tmp/data/ceval"

def download_medical_data():
    print("🚀 正在从 HF 镜像站拉取 C-Eval 医疗子集...")
    subjects = ["basic_medicine", "clinical_medicine", "physician"]
    
    # 自动创建目录
    os.makedirs(save_dir, exist_ok=True)
    
    for sub in subjects:
        try:
            # 加载特定的子科目验证集（含答案）
            ds = load_dataset("ceval/ceval-exam", name=sub, split="val", trust_remote_code=True)
            
            # 落地为物理文件
            file_path = os.path.join(save_dir, f"{sub}_val.csv")
            ds.to_pandas().to_csv(file_path, index=False)
            print(f"✅ {sub} 下载成功 -> {file_path}")
        except Exception as e:
            print(f"❌ {sub} 下载失败: {e}")

if __name__ == "__main__":
    download_medical_data()