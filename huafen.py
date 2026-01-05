import os
import random


def load_and_combine(file_paths):
    
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
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence, label in data:
            f.write(f"{sentence}\t{label}\n")
def main():
    input_files = [
        '/home/……/FLUTE/cleandata/MOH-X.txt',
        '/home/……/FLUTE/cleandata/TroFi-train.txt',
        '/home/……/FLUTE/cleandata/Metaphor_VUA_train.txt',
        '/home/……/FLUTE/cleandata/VUAverb-train.txt',
        '/home/……/FLUTE/cleandata/VUAverb-val.txt',
        '/home/……/FLUTE/cleandata/smile-train.txt',
        '/home/……/FLUTE/cleandata/relocalr-train.txt',
    ]

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, 

    all_data = load_and_combine(input_files)
    print(f"✅ 总共加载句子数: {len(all_data)}")

    train_data, val_data, test_data = shuffle_and_split(all_data, train_ratio, val_ratio)

    save_to_file(train_data, '/home/……/FLUTE/cleandata/train3.txt')
    save_to_file(val_data, '/home/……/FLUTE/cleandata/val3.txt')
    save_to_file(test_data, '/home/……/FLUTE/cleandata/test3.txt')

    print(f"✅ 训练集大小: {len(train_data)}")
    print(f"✅ 验证集大小: {len(val_data)}")
    print(f"✅ 测试集大小: {len(test_data)}")
    print("数据整合与划分完成")


if __name__ == '__main__':

    main()
