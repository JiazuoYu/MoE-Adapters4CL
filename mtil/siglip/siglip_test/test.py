import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

# 检查 GPU 可用性
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please ensure you have a GPU and the appropriate drivers installed.")

# 加载模型
model = AutoModel.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")

# 遍历模型的所有参数并打印名称
for name, _ in model.named_parameters():
    print(name)

# 加载处理器
processor = AutoProcessor.from_pretrained("/home/dhw/HZC_workspace/Models/siglip-base-patch16-224")
print("2")

# 加载图像
image = Image.open("/home/dhw/HZC_workspace/Projects/MoE-Adapters++/mtil/siglip/test/pipeline-cat-chonk.jpeg")
candidate_labels = ["a Pallas cat", "a lion", "a Siberian tiger"]
texts = [f'This is a photo of {label}.' for label in candidate_labels]
print("3")

# 指定设备为显卡 0
device = torch.device("cuda:0")

# 将模型移动到显卡 0
model = model.to(device)
print("Model device:", next(model.parameters()).device)  # 验证模型设备

# 预处理输入数据并移动到显卡 0
inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}  # 将所有输入张量移动到显卡 0
print("Input device:", inputs["pixel_values"].device)  # 验证输入数据设备
print("4")

# 执行模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取预测结果并计算概率
logits_per_image = outputs.logits_per_image
probs = torch.sigmoid(logits_per_image)
print(f"{probs[0][0]:.1%} that image 0 is '{candidate_labels[0]}'")