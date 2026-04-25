import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import warnings
import pandas as pd
import time
from tabulate import tabulate

# 设置绘图样式
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 禁用 KMeans 的 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

# 创建保存结果的目录
import os
os.makedirs('cifar10_results', exist_ok=True)

# CIFAR-10 类别名称
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 存储训练数据的全局变量
training_summary = {
    'dataset': 'CIFAR-10',
    'visual_words': 0,
    'phase_times': {},
    'final_accuracy': 0,
    'sample_counts': {}
}

# --- 1. 数据加载函数（完全保持原始逻辑）---
def load_and_extract_data(dataset_name='cifar10'):
    """Load CIFAR-10 dataset and return images and labels."""
    if dataset_name == 'cifar10':
        train_data = CIFAR10(root='./data', train=True, download=True)
        X_train = [np.array(img) for img in train_data.data]
        y_train = train_data.targets

        test_data = CIFAR10(root='./data', train=False, download=True)
        X_test = [np.array(img) for img in test_data.data]
        y_test = test_data.targets

        training_summary['sample_counts']['train'] = len(X_train)
        training_summary['sample_counts']['test'] = len(X_test)

        return X_train, y_train, X_test, y_test
    else:
        raise ValueError("Only 'cifar10' dataset is supported")

def quick_dataset_visualization(X, y):
    """快速查看数据集 - 修复版本"""
    # 确保y是numpy数组
    y_array = np.array(y)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()

    # 为每个类别找到样本
    for class_idx in range(10):
        # 在整个数据集中找该类别的第一个样本
        class_indices = np.where(y_array == class_idx)[0]

        if len(class_indices) > 0:
            # 使用该类别的第一个样本
            idx = class_indices[0]
            axes[class_idx].imshow(X[idx])
            axes[class_idx].set_title(f'Class {class_idx}: {cifar10_classes[class_idx]}')
        else:
            # 如果没有该类别，显示空白
            axes[class_idx].imshow(np.zeros((32, 32, 3), dtype=np.uint8))
            axes[class_idx].set_title(f'Class {class_idx}: Not found')

        axes[class_idx].axis('off')

    plt.suptitle('CIFAR-10 Dataset Samples (One per class)', fontsize=14)
    plt.tight_layout()
    plt.savefig('cifar10_results/dataset_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

# --- 2. SIFT 特征提取函数（完全保持原始逻辑）---
def extract_sift_descriptors(images, desc="Extracting SIFT Descriptors"):
    """Extract SIFT descriptors - 完全按照原始代码逻辑"""
    sift = cv2.SIFT_create()
    all_descriptors = []
    image_descriptors_list = []
    valid_indices = []

    for i, img in enumerate(tqdm(images, desc=desc)):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, des = sift.detectAndCompute(gray_img, None)

        if des is not None and des.shape[0] > 0:
            all_descriptors.append(des)
            image_descriptors_list.append(des)
            valid_indices.append(i)
        else:
            image_descriptors_list.append(None)

    if all_descriptors:
        all_descriptors_concatenated = np.concatenate(all_descriptors)
    else:
        all_descriptors_concatenated = np.array([])

    return image_descriptors_list, all_descriptors_concatenated, valid_indices

# --- 3. 词袋模型函数（完全保持原始逻辑）---
def create_bow_features(image_descriptors, kmeans_model):
    """Convert SIFT descriptors to BoW histogram features."""
    k = kmeans_model.n_clusters
    image_features = []

    for des in image_descriptors:
        labels = kmeans_model.predict(des)
        hist, _ = np.histogram(labels, bins=k, range=(0, k))
        image_features.append(hist)

    return np.array(image_features)

# --- 4. 主运行逻辑（完全保持原始流程和参数）---
def run_traditional_cv(dataset_name, visual_words_k):
    """主要运行函数 - 完全保持您原始代码的流程和参数"""
    training_summary['visual_words'] = visual_words_k
    training_summary['phase_times'] = {}

    print(f"\n{'='*80}")
    print(f"RUNNING SIFT + BoW + SVM on {dataset_name.upper()} with K={visual_words_k}")
    print(f"{'='*80}\n")

    # 1. 加载数据
    print("1. Loading Dataset...")
    start_time = time.time()
    X_train, y_train, X_test, y_test = load_and_extract_data(dataset_name)
    training_summary['phase_times']['data_loading'] = time.time() - start_time
    print(f"   Training set: {len(X_train)} images")
    print(f"   Test set: {len(X_test)} images")

    # 快速可视化 - 使用整个数据集查找
    print("   Visualizing dataset samples...")
    quick_dataset_visualization(X_train, y_train)

    # --- 训练数据处理 ---
    # 2. 提取 SIFT 描述符
    print("\n2. Extracting SIFT descriptors from training set...")
    start_time = time.time()
    train_descriptors_list, all_train_descriptors, train_valid_indices = extract_sift_descriptors(X_train)
    training_summary['phase_times']['sift_train'] = time.time() - start_time

    # 3. 过滤无效样本
    y_train_filtered = np.array(y_train)[train_valid_indices].tolist()
    train_descriptors_list_filtered = [train_descriptors_list[i] for i in train_valid_indices]

    if len(y_train_filtered) == 0:
        print("Error: No SIFT descriptors could be extracted from the training set.")
        return

    print(f"   Valid training samples after SIFT extraction: {len(y_train_filtered)}")

    # 4. 训练 K-Means 模型 - 完全使用您原始的参数
    print(f"\n3. Training K-Means with K={visual_words_k}...")
    print("   Note: Using original parameters - n_clusters=K, random_state=42, n_init=10, verbose=0")
    start_time = time.time()

    # 完全按照您原始代码的参数
    kmeans = MiniBatchKMeans(
        n_clusters=visual_words_k,
        random_state=42,
        n_init=10,
        verbose=0
        # 不添加任何额外参数
    )
    kmeans.fit(all_train_descriptors)
    training_summary['phase_times']['kmeans'] = time.time() - start_time
    print(f"   K-Means training completed in {training_summary['phase_times']['kmeans']:.2f}s")

    # 5. 创建训练集 BoW 特征
    print("\n4. Creating BoW features for training set...")
    start_time = time.time()
    X_train_bow = create_bow_features(train_descriptors_list_filtered, kmeans)
    training_summary['phase_times']['bow_train'] = time.time() - start_time
    print(f"   BoW feature dimension: {X_train_bow.shape}")

    # --- 测试数据处理 ---
    print("\n5. Processing test set...")
    start_time = time.time()
    test_descriptors_list, _, test_valid_indices = extract_sift_descriptors(X_test)
    training_summary['phase_times']['sift_test'] = time.time() - start_time

    y_test_filtered = np.array(y_test)[test_valid_indices].tolist()
    test_descriptors_list_filtered = [test_descriptors_list[i] for i in test_valid_indices]

    X_test_bow = create_bow_features(test_descriptors_list_filtered, kmeans)

    print(f"   Valid test samples after SIFT extraction: {len(y_test_filtered)}")

    # 6. 特征标准化
    print("\n6. Standardizing features...")
    start_time = time.time()
    scaler = StandardScaler().fit(X_train_bow)
    X_train_scaled = scaler.transform(X_train_bow)
    X_test_scaled = scaler.transform(X_test_bow)
    training_summary['phase_times']['scaling'] = time.time() - start_time

    # 7. 训练 SVC 分类器 - 完全使用您原始的参数
    print("\n7. Training SVC Classifier (Slow, please wait)...")
    print(f"   Final Training Samples: X={X_train_scaled.shape[0]}, Y={len(y_train_filtered)}")
    print("   Note: Using original parameters - kernel='linear', C=1.0, random_state=42")

    start_time = time.time()

    # 完全按照您原始代码的参数
    svm = SVC(
        kernel='linear',
        C=1.0,
        random_state=42
        # 不添加任何额外参数
    )
    svm.fit(X_train_scaled, y_train_filtered)
    training_summary['phase_times']['svm_training'] = time.time() - start_time

    print(f"   SVC training completed in {training_summary['phase_times']['svm_training']:.2f}s")

    # 8. 评估
    print("\n8. Evaluating model...")
    y_pred = svm.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test_filtered, y_pred) * 100
    training_summary['final_accuracy'] = test_accuracy

    print(f"   Evaluation performed on {len(y_test_filtered)} test samples.")
    print(f"   Overall Test Accuracy: {test_accuracy:.2f}%")

    # 9. 生成可视化结果
    generate_visualizations(y_test_filtered, y_pred)

    # 10. 打印结果表格
    print_results_summary()

    # 11. 打印分类报告
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test_filtered, y_pred, target_names=cifar10_classes, zero_division=0))

def generate_visualizations(y_true, y_pred):
    """生成必要的可视化图表"""
    print("\n9. Generating visualizations...")

    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(len(cifar10_classes))
    plt.xticks(tick_marks, cifar10_classes, rotation=45)
    plt.yticks(tick_marks, cifar10_classes)

    # 在格子中添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('cifar10_results/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 2. 时间分析图表
    phases = list(training_summary['phase_times'].keys())
    times = list(training_summary['phase_times'].values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(phases)), times, color='skyblue')
    plt.title('Processing Time per Phase', fontsize=14, fontweight='bold')
    plt.xlabel('Processing Phase')
    plt.ylabel('Time (seconds)')
    plt.xticks(range(len(phases)), phases, rotation=45, ha='right')

    # 在柱子上添加时间数值
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('cifar10_results/processing_time.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 3. 类别准确率
    class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(10), class_accuracy, color='lightgreen', edgecolor='black')
    plt.title('Class-wise Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(10), [f'{i}\n{cifar10_classes[i]}' for i in range(10)])
    plt.ylim([0, 100])

    # 在柱子上添加准确率数值
    for bar, acc in zip(bars, class_accuracy):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('cifar10_results/class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_results_summary():
    """打印结果摘要表格"""
    print("\n" + "="*80)
    print("RESULTS SUMMARY TABLE")
    print("="*80)

    # 准备表格数据
    table_data = []
    total_time = 0

    # 先计算总时间
    for phase_time in training_summary['phase_times'].values():
        total_time += phase_time

    # 生成表格行
    for phase, phase_time in training_summary['phase_times'].items():
        percentage = (phase_time / total_time) * 100 if total_time > 0 else 0
        table_data.append([
            phase.replace('_', ' ').title(),
            f"{phase_time:.2f}s",
            f"{percentage:.1f}%"
        ])

    # 添加总计行
    table_data.append([
        "TOTAL",
        f"{total_time:.2f}s",
        "100.0%"
    ])

    print(tabulate(table_data,
                  headers=['Phase', 'Time Taken', 'Percentage of Total'],
                  tablefmt='grid'))

    print(f"\nDataset Statistics:")
    print(f"  - Dataset: {training_summary['dataset']}")
    print(f"  - Visual Vocabulary Size: {training_summary['visual_words']}")
    print(f"  - Training Samples: {training_summary['sample_counts']['train']}")
    print(f"  - Test Samples: {training_summary['sample_counts']['test']}")
    print(f"  - Final Test Accuracy: {training_summary['final_accuracy']:.2f}%")
    print(f"  - Total Processing Time: {total_time/60:.1f} minutes")

    # 保存详细结果到CSV
    save_to_csv(total_time)

def save_to_csv(total_time):
    """保存结果到CSV文件"""
    results_df = pd.DataFrame({
        'Dataset': [training_summary['dataset']],
        'Visual_Words': [training_summary['visual_words']],
        'Final_Accuracy': [training_summary['final_accuracy']],
        'Total_Time_Seconds': [total_time],
        'Total_Time_Minutes': [total_time/60],
        'KMeans_n_init': [10],  # 您的原始参数
        'SVM_kernel': ['linear'],  # 您的原始参数
        'SVM_C': [1.0]  # 您的原始参数
    })

    # 添加各阶段时间
    for phase, phase_time in training_summary['phase_times'].items():
        results_df[f'Time_{phase}'] = phase_time

    results_df.to_csv('cifar10_results/training_results.csv', index=False)
    print(f"\nDetailed results saved to: cifar10_results/training_results.csv")

# --- 6. 运行程序 ---
if __name__ == '__main__':
    VISUAL_WORDS_K = 200

    print("CIFAR-10 DATASET PROCESSING")
    print("="*50)
    print("Note: Using EXACTLY the same parameters as the original code:")
    print("  - KMeans: n_clusters=K, random_state=42, n_init=10, verbose=0")
    print("  - SVM: kernel='linear', C=1.0, random_state=42")
    print("="*50)

    try:
        accuracy = run_traditional_cv('cifar10', VISUAL_WORDS_K)
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()