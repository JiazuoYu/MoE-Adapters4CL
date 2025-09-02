import os
import torch
from torch.utils.data import Dataset
from PIL import Image

from .collections import ClassificationDataset


class BaseDomainet(ClassificationDataset):
    def __init__(
        self,
        domain: str,  # 新增domain参数
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        super().__init__(
            preprocess,
            location=os.path.join(location, domain),  # 动态构建路径
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_workers=num_workers,
            append_dataset_name_to_template=append_dataset_name_to_template,
        )
        
        self.name = f"domainet_{domain}"
        self.domain = domain
        self.classnames = self._load_class_info()
        self.process_labels()
        
        # 构建训练和测试数据集
        self.train_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_train.txt"))
        self.test_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_test.txt"))
        self.build_dataloader()
        
        # 统一模板配置
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]

    def _load_class_info(self):
        """从所有数据中收集类别信息并验证一致性"""
        label_to_class = {}
        seen_classes = set()
        
        # 检查训练集和测试集
        for split in [f"{self.domain}_train.txt", f"{self.domain}_test.txt"]:
            with open(os.path.join(self.location, split)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析路径和标签
                    img_rel_path, label = line.rsplit(" ", 1)
                    label = int(label)
                    
                    # 从路径获取类别名称
                    class_name = img_rel_path.split("/")[1]  # 假设格式为xxx/class_name/xxx.jpg
                    
                    # 验证标签一致性
                    if label in label_to_class:
                        if label_to_class[label] != class_name:
                            raise ValueError(f"Label {label} 对应多个类别: {label_to_class[label]} 和 {class_name}")
                    else:
                        # 验证类别唯一性
                        if class_name in seen_classes:
                            raise ValueError(f"类别 {class_name} 出现在多个标签中")
                        label_to_class[label] = class_name
                        seen_classes.add(class_name)
        
        # 验证标签连续性
        max_label = max(label_to_class.keys())
        for label in range(max_label + 1):
            if label not in label_to_class:
                raise ValueError(f"标签不连续，缺少标签 {label}")
        
        return [label_to_class[label] for label in sorted(label_to_class)]

    def _create_dataset(self, split_file):
        """创建单个数据集"""
        samples = []
        
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_rel_path, label = line.rsplit(" ", 1)
                label = int(label)
                full_path = os.path.join(self.location, img_rel_path)
                
                # 验证路径存在
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"图片不存在: {full_path}")
                
                samples.append((full_path, label))
        
        return CustomImageDataset(
            samples=samples,
            preprocess=self.preprocess,
            classnames=self.classnames
        )

# # 自动生成子类
# domain_classes = {
    # "ClipArt": "clipart",
    # "InfoGraph": "infograph",
    # "Painting": "painting",
    # "QuickDraw": "quickdraw",
    # "Real": "real",
    # "Sketch": "sketch"
# }

# for cls_name, domain in domain_classes.items():
#     globals()[cls_name] = type(
#         cls_name,
#         (BaseDomainet,),
#         {"__init__": lambda self, preprocess, location=os.path.expanduser("./data"), **kwargs: 
#             super(self.__class__, self).__init__(
#                 domain=domain,
#                 preprocess=preprocess,
#                 location=location,
#                 **kwargs
#             )}
#     )

class Real(ClassificationDataset):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        super().__init__(
            preprocess,
            location=location,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_workers=num_workers,
            append_dataset_name_to_template=append_dataset_name_to_template,
        )
        
        self.name = "domainet"
        self.domain = "real"
        self.location = f"{location}/{self.domain}"
        self.classnames = self._load_class_info()
        # print(self.classnames)
        self.process_labels()
        
        # 构建训练和测试数据集
        self.train_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_train.txt"))
        # print(len(self.train_dataset))
        self.test_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_test.txt"))
        self.build_dataloader()
        
        # 补充模板部分（核心实现）
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
        
    def _load_class_info(self):
        """从所有数据中收集类别信息并验证一致性"""
        label_to_class = {}
        seen_classes = set()
        
        # 检查训练集和测试集
        for split in [f"{self.domain}_train.txt", f"{self.domain}_test.txt"]:
            with open(os.path.join(self.location, split)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析路径和标签
                    img_rel_path, label = line.rsplit(" ", 1)
                    label = int(label)
                    
                    # 从路径获取类别名称
                    class_name = img_rel_path.split("/")[1]  # 假设格式为xxx/class_name/xxx.jpg
                    
                    # 验证标签一致性
                    if label in label_to_class:
                        if label_to_class[label] != class_name:
                            raise ValueError(f"Label {label} 对应多个类别: {label_to_class[label]} 和 {class_name}")
                    else:
                        # 验证类别唯一性
                        if class_name in seen_classes:
                            raise ValueError(f"类别 {class_name} 出现在多个标签中")
                        label_to_class[label] = class_name
                        seen_classes.add(class_name)
        
        # 验证标签连续性
        max_label = max(label_to_class.keys())
        for label in range(max_label + 1):
            if label not in label_to_class:
                raise ValueError(f"标签不连续，缺少标签 {label}")
        
        return [label_to_class[label] for label in sorted(label_to_class)]

    def _create_dataset(self, split_file):
        """创建单个数据集"""
        samples = []
        
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_rel_path, label = line.rsplit(" ", 1)
                label = int(label)
                full_path = os.path.join(self.location, img_rel_path)
                
                # 验证路径存在
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"图片不存在: {full_path}")
                
                samples.append((full_path, label))
        
        return CustomImageDataset(
            samples=samples,
            preprocess=self.preprocess,
            classnames=self.classnames
        )


class ClipArt(ClassificationDataset):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        super().__init__(
            preprocess,
            location=location,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_workers=num_workers,
            append_dataset_name_to_template=append_dataset_name_to_template,
        )
        
        self.name = "domainet"
        self.domain = "clipart"
        self.location = f"{location}/{self.domain}"
        self.classnames = self._load_class_info()
        # print(self.classnames)
        self.process_labels()
        
        # 构建训练和测试数据集
        self.train_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_train.txt"))
        # print(len(self.train_dataset))
        self.test_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_test.txt"))
        self.build_dataloader()
        
        # 补充模板部分（核心实现）
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
        
    def _load_class_info(self):
        """从所有数据中收集类别信息并验证一致性"""
        label_to_class = {}
        seen_classes = set()
        
        # 检查训练集和测试集
        for split in [f"{self.domain}_train.txt", f"{self.domain}_test.txt"]:
            with open(os.path.join(self.location, split)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析路径和标签
                    img_rel_path, label = line.rsplit(" ", 1)
                    label = int(label)
                    
                    # 从路径获取类别名称
                    class_name = img_rel_path.split("/")[1]  # 假设格式为xxx/class_name/xxx.jpg
                    
                    # 验证标签一致性
                    if label in label_to_class:
                        if label_to_class[label] != class_name:
                            raise ValueError(f"Label {label} 对应多个类别: {label_to_class[label]} 和 {class_name}")
                    else:
                        # 验证类别唯一性
                        if class_name in seen_classes:
                            raise ValueError(f"类别 {class_name} 出现在多个标签中")
                        label_to_class[label] = class_name
                        seen_classes.add(class_name)
        
        # 验证标签连续性
        max_label = max(label_to_class.keys())
        for label in range(max_label + 1):
            if label not in label_to_class:
                raise ValueError(f"标签不连续，缺少标签 {label}")
        
        return [label_to_class[label] for label in sorted(label_to_class)]

    def _create_dataset(self, split_file):
        """创建单个数据集"""
        samples = []
        
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_rel_path, label = line.rsplit(" ", 1)
                label = int(label)
                full_path = os.path.join(self.location, img_rel_path)
                
                # 验证路径存在
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"图片不存在: {full_path}")
                
                samples.append((full_path, label))
        
        return CustomImageDataset(
            samples=samples,
            preprocess=self.preprocess,
            classnames=self.classnames
        )


class InfoGraph(ClassificationDataset):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        super().__init__(
            preprocess,
            location=location,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_workers=num_workers,
            append_dataset_name_to_template=append_dataset_name_to_template,
        )
        
        self.name = "domainet"
        self.domain = "infograph"
        self.location = f"{location}/{self.domain}"
        self.classnames = self._load_class_info()
        # print(self.classnames)
        self.process_labels()
        
        # 构建训练和测试数据集
        self.train_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_train.txt"))
        # print(len(self.train_dataset))
        self.test_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_test.txt"))
        self.build_dataloader()
        
        # 补充模板部分（核心实现）
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
        
    def _load_class_info(self):
        """从所有数据中收集类别信息并验证一致性"""
        label_to_class = {}
        seen_classes = set()
        
        # 检查训练集和测试集
        for split in [f"{self.domain}_train.txt", f"{self.domain}_test.txt"]:
            with open(os.path.join(self.location, split)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析路径和标签
                    img_rel_path, label = line.rsplit(" ", 1)
                    label = int(label)
                    
                    # 从路径获取类别名称
                    class_name = img_rel_path.split("/")[1]  # 假设格式为xxx/class_name/xxx.jpg
                    
                    # 验证标签一致性
                    if label in label_to_class:
                        if label_to_class[label] != class_name:
                            raise ValueError(f"Label {label} 对应多个类别: {label_to_class[label]} 和 {class_name}")
                    else:
                        # 验证类别唯一性
                        if class_name in seen_classes:
                            raise ValueError(f"类别 {class_name} 出现在多个标签中")
                        label_to_class[label] = class_name
                        seen_classes.add(class_name)
        
        # 验证标签连续性
        max_label = max(label_to_class.keys())
        for label in range(max_label + 1):
            if label not in label_to_class:
                raise ValueError(f"标签不连续，缺少标签 {label}")
        
        return [label_to_class[label] for label in sorted(label_to_class)]

    def _create_dataset(self, split_file):
        """创建单个数据集"""
        samples = []
        
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_rel_path, label = line.rsplit(" ", 1)
                label = int(label)
                full_path = os.path.join(self.location, img_rel_path)
                
                # 验证路径存在
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"图片不存在: {full_path}")
                
                samples.append((full_path, label))
        
        return CustomImageDataset(
            samples=samples,
            preprocess=self.preprocess,
            classnames=self.classnames
        )


class Painting(ClassificationDataset):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        super().__init__(
            preprocess,
            location=location,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_workers=num_workers,
            append_dataset_name_to_template=append_dataset_name_to_template,
        )
        
        self.name = "domainet"
        self.domain = "painting"
        self.location = f"{location}/{self.domain}"
        self.classnames = self._load_class_info()
        # print(self.classnames)
        self.process_labels()
        
        # 构建训练和测试数据集
        self.train_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_train.txt"))
        # print(len(self.train_dataset))
        self.test_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_test.txt"))
        self.build_dataloader()
        
        # 补充模板部分（核心实现）
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
        
    def _load_class_info(self):
        """从所有数据中收集类别信息并验证一致性"""
        label_to_class = {}
        seen_classes = set()
        
        # 检查训练集和测试集
        for split in [f"{self.domain}_train.txt", f"{self.domain}_test.txt"]:
            with open(os.path.join(self.location, split)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析路径和标签
                    img_rel_path, label = line.rsplit(" ", 1)
                    label = int(label)
                    
                    # 从路径获取类别名称
                    class_name = img_rel_path.split("/")[1]  # 假设格式为xxx/class_name/xxx.jpg
                    
                    # 验证标签一致性
                    if label in label_to_class:
                        if label_to_class[label] != class_name:
                            raise ValueError(f"Label {label} 对应多个类别: {label_to_class[label]} 和 {class_name}")
                    else:
                        # 验证类别唯一性
                        if class_name in seen_classes:
                            raise ValueError(f"类别 {class_name} 出现在多个标签中")
                        label_to_class[label] = class_name
                        seen_classes.add(class_name)
        
        # 验证标签连续性
        max_label = max(label_to_class.keys())
        for label in range(max_label + 1):
            if label not in label_to_class:
                raise ValueError(f"标签不连续，缺少标签 {label}")
        
        return [label_to_class[label] for label in sorted(label_to_class)]

    def _create_dataset(self, split_file):
        """创建单个数据集"""
        samples = []
        
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_rel_path, label = line.rsplit(" ", 1)
                label = int(label)
                full_path = os.path.join(self.location, img_rel_path)
                
                # 验证路径存在
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"图片不存在: {full_path}")
                
                samples.append((full_path, label))
        
        return CustomImageDataset(
            samples=samples,
            preprocess=self.preprocess,
            classnames=self.classnames
        )


class QuickDraw(ClassificationDataset):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        super().__init__(
            preprocess,
            location=location,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_workers=num_workers,
            append_dataset_name_to_template=append_dataset_name_to_template,
        )
        
        self.name = "domainet"
        self.domain = "quickdraw"
        self.location = f"{location}/{self.domain}"
        self.classnames = self._load_class_info()
        # print(self.classnames)
        self.process_labels()
        
        # 构建训练和测试数据集
        self.train_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_train.txt"))
        # print(len(self.train_dataset))
        self.test_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_test.txt"))
        self.build_dataloader()
        
        # 补充模板部分（核心实现）
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
        
    def _load_class_info(self):
        """从所有数据中收集类别信息并验证一致性"""
        label_to_class = {}
        seen_classes = set()
        
        # 检查训练集和测试集
        for split in [f"{self.domain}_train.txt", f"{self.domain}_test.txt"]:
            with open(os.path.join(self.location, split)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析路径和标签
                    img_rel_path, label = line.rsplit(" ", 1)
                    label = int(label)
                    
                    # 从路径获取类别名称
                    class_name = img_rel_path.split("/")[1]  # 假设格式为xxx/class_name/xxx.jpg
                    
                    # 验证标签一致性
                    if label in label_to_class:
                        if label_to_class[label] != class_name:
                            raise ValueError(f"Label {label} 对应多个类别: {label_to_class[label]} 和 {class_name}")
                    else:
                        # 验证类别唯一性
                        if class_name in seen_classes:
                            raise ValueError(f"类别 {class_name} 出现在多个标签中")
                        label_to_class[label] = class_name
                        seen_classes.add(class_name)
        
        # 验证标签连续性
        max_label = max(label_to_class.keys())
        for label in range(max_label + 1):
            if label not in label_to_class:
                raise ValueError(f"标签不连续，缺少标签 {label}")
        
        return [label_to_class[label] for label in sorted(label_to_class)]

    def _create_dataset(self, split_file):
        """创建单个数据集"""
        samples = []
        
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_rel_path, label = line.rsplit(" ", 1)
                label = int(label)
                full_path = os.path.join(self.location, img_rel_path)
                
                # 验证路径存在
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"图片不存在: {full_path}")
                
                samples.append((full_path, label))
        
        return CustomImageDataset(
            samples=samples,
            preprocess=self.preprocess,
            classnames=self.classnames
        )


class Sketch(ClassificationDataset):
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("./data"),
        batch_size=128,
        batch_size_eval=None,
        num_workers=16,
        append_dataset_name_to_template=False,
    ) -> None:
        super().__init__(
            preprocess,
            location=location,
            batch_size=batch_size,
            batch_size_eval=batch_size_eval,
            num_workers=num_workers,
            append_dataset_name_to_template=append_dataset_name_to_template,
        )
        
        self.name = "domainet"
        self.domain = "sketch"
        self.location = f"{location}/{self.domain}"
        self.classnames = self._load_class_info()
        # print(self.classnames)
        self.process_labels()
        
        # 构建训练和测试数据集
        self.train_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_train.txt"))
        # print(len(self.train_dataset))
        self.test_dataset = self._create_dataset(os.path.join(self.location, f"{self.domain}_test.txt"))
        self.build_dataloader()
        
        # 补充模板部分（核心实现）
        self.templates = [
            lambda c: f"a photo of a {c}.",
            lambda c: f"a photo of the {c}.",
        ]
        
    def _load_class_info(self):
        """从所有数据中收集类别信息并验证一致性"""
        label_to_class = {}
        seen_classes = set()
        
        # 检查训练集和测试集
        for split in [f"{self.domain}_train.txt", f"{self.domain}_test.txt"]:
            with open(os.path.join(self.location, split)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 解析路径和标签
                    img_rel_path, label = line.rsplit(" ", 1)
                    label = int(label)
                    
                    # 从路径获取类别名称
                    class_name = img_rel_path.split("/")[1]  # 假设格式为xxx/class_name/xxx.jpg
                    
                    # 验证标签一致性
                    if label in label_to_class:
                        if label_to_class[label] != class_name:
                            raise ValueError(f"Label {label} 对应多个类别: {label_to_class[label]} 和 {class_name}")
                    else:
                        # 验证类别唯一性
                        if class_name in seen_classes:
                            raise ValueError(f"类别 {class_name} 出现在多个标签中")
                        label_to_class[label] = class_name
                        seen_classes.add(class_name)
        
        # 验证标签连续性
        max_label = max(label_to_class.keys())
        for label in range(max_label + 1):
            if label not in label_to_class:
                raise ValueError(f"标签不连续，缺少标签 {label}")
        
        return [label_to_class[label] for label in sorted(label_to_class)]

    def _create_dataset(self, split_file):
        """创建单个数据集"""
        samples = []
        
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                img_rel_path, label = line.rsplit(" ", 1)
                label = int(label)
                full_path = os.path.join(self.location, img_rel_path)
                
                # 验证路径存在
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"图片不存在: {full_path}")
                
                samples.append((full_path, label))
        
        return CustomImageDataset(
            samples=samples,
            preprocess=self.preprocess,
            classnames=self.classnames
        )


class CustomImageDataset(Dataset):
    def __init__(self, samples, preprocess, classnames):
        self.samples = samples
        self.preprocess = preprocess
        self.classnames = classnames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 加载图像
        img = Image.open(img_path).convert("RGB")
        if self.preprocess:
            img = self.preprocess(img)
        
        # 验证标签有效性
        if label < 0 or label >= len(self.classnames):
            raise ValueError(f"无效标签 {label}，类别总数 {len(self.classnames)}")
        
        return img, label

    def get_classname(self, label):
        """通过标签获取类别名称"""
        return self.classnames[label]