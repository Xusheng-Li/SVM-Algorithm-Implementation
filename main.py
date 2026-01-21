import numpy as np
import matplotlib.pyplot as plt
# 全局中文字体配置
import matplotlib
from matplotlib.font_manager import FontProperties

# 使用绝对路径指定字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # Windows系统下SimHei字体的标准路径
font_prop = FontProperties(fname=font_path)
matplotlib.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Heiti TC',
                                   'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 添加日志过滤器忽略字体警告
import logging

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import pandas as pd


# 加载MNIST数据集
def load_mnist():
    """加载MNIST数据集并进行预处理，手写字数据集"""
    print("正在加载MNIST数据集...")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X, y = mnist['data'], mnist['target']

    # 转换数据类型
    X = X.astype(np.float64)
    y = y.astype(np.int64)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# 训练SVM模型并评估性能
def train_and_evaluate_svm(X_train, X_test, y_train, y_test, kernel, gamma='scale'):
    """
    训练SVM模型并评估性能

    参数:
        X_train, X_test: 训练集和测试集特征
        y_train, y_test: 训练集和测试集标签
        kernel: 核函数类型
        gamma: 高斯核参数
    """
    print(f"\n正在使用{kernel}核训练SVM模型 (gamma={gamma})...")
    start_time = time.time()

    # 创建SVM分类器
    svm = SVC(kernel=kernel, gamma=gamma, C=1.0, random_state=42)

    # 训练模型
    svm.fit(X_train, y_train)

    # 评估模型
    train_time = time.time() - start_time
    y_pred_train = svm.predict(X_train)
    y_pred_test = svm.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"训练时间: {train_time:.2f}秒")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    return {
        'kernel': kernel,
        'gamma': gamma,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_time': train_time,
        'model': svm,
        'y_pred_test': y_pred_test
    }


# 研究不同gamma值对RBF核性能的影响
def explore_gamma_effect(X_train, X_test, y_train, y_test, gamma_values):
    """
    研究不同gamma值对RBF核SVM性能的影响

    参数:
        gamma_values: 要测试的gamma值列表
    """
    results = []

    for gamma in gamma_values:
        result = train_and_evaluate_svm(X_train, X_test, y_train, y_test, 'rbf', gamma)
        results.append(result)

    # 可视化不同gamma值的性能
    plt.figure(figsize=(10, 6))
    plt.plot([r['gamma'] for r in results], [r['test_accuracy'] for r in results], 'o-', label='测试集准确率')
    plt.plot([r['gamma'] for r in results], [r['train_accuracy'] for r in results], 's-', label='训练集准确率')
    plt.xscale('log')
    plt.xlabel('gamma值')
    plt.ylabel('准确率')
    plt.title('RBF核SVM: gamma值对性能的影响')
    plt.grid(True)
    plt.legend()
    plt.savefig('gamma_effect.png')
    plt.show(block=True)  # 确保绘图窗口保持打开

    # 输出结果表格
    print("\n不同gamma值的性能对比:")
    results_df = pd.DataFrame(results)
    print(results_df[['gamma', 'train_accuracy', 'test_accuracy', 'train_time']])

    return results


# 可视化混淆矩阵
def visualize_confusion_matrix(y_true, y_pred, title):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))

    # 在混淆矩阵上标注数值
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show(block=True)  # 确保绘图窗口保持打开


# 可视化错误分类的样本
def visualize_misclassified(X_test, y_test, y_pred, kernel, gamma='scale', num_samples=12):
    """可视化错误分类的样本"""
    # 确保所有数据都是numpy数组格式
    y_test_np = y_test.to_numpy() if hasattr(y_test, 'to_numpy') else y_test
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else y_pred.to_numpy()

    misclassified = (y_test_np != y_pred_np)
    X_mis = X_test[misclassified]
    y_true_mis = y_test_np[misclassified]
    y_pred_mis = y_pred_np[misclassified]

    if len(X_mis) == 0:
        print("没有错误分类的样本!")
        return

    plt.figure(figsize=(12, 8))
    for i in range(min(num_samples, len(X_mis))):
        plt.subplot(3, 4, i + 1)
        plt.imshow(X_mis[i].reshape(28, 28), cmap='gray')
        # 修改这里：将.iloc[i]改为[i]
        plt.title(f'真实: {y_true_mis[i]}, 预测: {y_pred_mis[i]}')
        plt.axis('off')
    plt.suptitle(f'{kernel}核SVM错误分类的样本 (gamma={gamma})')
    plt.tight_layout()
    plt.savefig(f'misclassified_{kernel}_gamma_{gamma}.png')
    print(f"已保存错误分类样本图: misclassified_{kernel}_gamma_{gamma}.png")
    plt.show(block=True)  # 确保绘图窗口保持打开
    plt.close()  # 显式关闭当前图形


# 主函数
def main():
    # 加载数据
    X_train, X_test, y_train, y_test = load_mnist()

    # 限制训练集大小以加快实验速度
    max_train_samples = 10000
    if len(X_train) > max_train_samples:
        X_train = X_train[:max_train_samples]
        y_train = y_train[:max_train_samples]
        print(f"使用{max_train_samples}个样本进行训练以加快速度")

    # 训练线性核SVM
    linear_result = train_and_evaluate_svm(X_train, X_test, y_train, y_test, 'linear')

    # 可视化线性核的混淆矩阵
    visualize_confusion_matrix(y_test, linear_result['y_pred_test'], '线性核SVM混淆矩阵')

    # 可视化线性核的错误分类样本
    visualize_misclassified(X_test, y_test, linear_result['y_pred_test'], 'linear')

    # 研究不同gamma值对RBF核的影响
    gamma_values = [0.0001, 0.001, 0.01, 0.1, 1.0]
    rbf_results = explore_gamma_effect(X_train, X_test, y_train, y_test, gamma_values)

    # 找出最佳gamma值的模型
    best_rbf_result = max(rbf_results, key=lambda x: x['test_accuracy'])
    print(f"\n最佳RBF核模型: gamma={best_rbf_result['gamma']}, 测试集准确率={best_rbf_result['test_accuracy']:.4f}")

    # 可视化最佳RBF核的混淆矩阵
    visualize_confusion_matrix(y_test, best_rbf_result['y_pred_test'],
                               f'RBF核SVM混淆矩阵 (gamma={best_rbf_result["gamma"]})')

    # 可视化最佳RBF核的错误分类样本
    visualize_misclassified(X_test, y_test, best_rbf_result['y_pred_test'],
                            'rbf', best_rbf_result['gamma'])

    # 性能对比表格
    print("\n线性核与最佳RBF核性能对比:")
    performance_comparison = pd.DataFrame([
        {
            '核函数': '线性',
            'gamma': 'N/A',
            '测试集准确率': linear_result['test_accuracy'],
            '训练时间(秒)': linear_result['train_time']
        },
        {
            '核函数': 'RBF',
            'gamma': best_rbf_result['gamma'],
            '测试集准确率': best_rbf_result['test_accuracy'],
            '训练时间(秒)': best_rbf_result['train_time']
        }
    ])
    print(performance_comparison)


if __name__ == "__main__":
    main()
    # 保持所有绘图窗口打开
    plt.show()