import os
import random


def load_and_combine(file_paths):
    """从多个文件中加载并合并句子-标签数据"""
    all_data = []
    for path in file_paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) != 2:
                    continue
                sentence, label = parts
                if label not in {'0', '1', '2', '3'}:
                    continue
                all_data.append((sentence.strip(), label))
    return all_data


def shuffle_and_split(data, train_ratio=0.8, val_ratio=0.1, seed=42):
    """打乱数据并按比例划分为 train / val / test"""
    random.seed(seed)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def save_to_file(data, output_path):
    """将数据保存为文件，每行格式为 sentence<TAB>label"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence, label in data:
            f.write(f"{sentence}\t{label}\n")
def main():
    # ✏️ 修改为你的输入文件路径
    input_files = [
        '/home/g23tka23/WSD/cleandata/MOH-X.txt',
        '/home/g23tka23/WSD/cleandata/TroFi-train.txt',
        '/home/g23tka23/WSD/cleandata/Metaphor_VUA_train.txt',
        '/home/g23tka23/WSD/cleandata/VUAverb-train.txt',
        '/home/g23tka23/WSD/cleandata/VUAverb-val.txt',
        '/home/g23tka23/WSD/cleandata/smile-train.txt',
        '/home/g23tka23/WSD/cleandata/relocalr-train.txt',
        # 可以继续添加
    ]

    # ✅ 设置比例（总和不超过 1.0）
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "划分比例总和必须为1"

    all_data = load_and_combine(input_files)
    print(f"✅ 总共加载句子数: {len(all_data)}")

    train_data, val_data, test_data = shuffle_and_split(all_data, train_ratio, val_ratio)

    save_to_file(train_data, '/home/g23tka23/WSD/cleandata/train3.txt')
    save_to_file(val_data, '/home/g23tka23/WSD/cleandata/val3.txt')
    save_to_file(test_data, '/home/g23tka23/WSD/cleandata/test3.txt')

    print(f"✅ 训练集大小: {len(train_data)}")
    print(f"✅ 验证集大小: {len(val_data)}")
    print(f"✅ 测试集大小: {len(test_data)}")
    print("🎉 数据整合与划分完成！")


if __name__ == '__main__':
    main()