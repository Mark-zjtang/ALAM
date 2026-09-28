import os
import re
from collections import Counter

def analyze_gif_files(folder_path):
    """
    分析文件夹中的gif文件，根据命名规则统计值
    
    文件命名格式: 数字1-数字2-字符串1-success/fail.gif
    - 数字1: 0-4
    - 数字2: 0-20  
    - 如果为fail: 统计 数字1-1 的值
    - 如果为success且数字1为4: 统计4
    - 其他情况忽略
    """
    
    # 正则表达式匹配文件名格式
    pattern = r'^(\d+)-(\d+)-(.+)-(succ|fail)\.gif$'
    
    values = []
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.gif'):
            match = re.match(pattern, filename)
            if match:
                digit1 = int(match.group(1))  # 数字1
                digit2 = int(match.group(2))  # 数字2
                string1 = match.group(3)      # 字符串1
                status = match.group(4)       # success/fail
                
                # 验证数字范围
                if 0 <= digit2 <= 4 and 0 <= digit1 <= 50:
                    if status == 'fail':
                        # fail时统计 数字1-1 的值
                        values.append(digit2 - 1)
                    elif status == 'succ' and digit2 == 4:
                        # success且数字1为4时统计4
                        values.append(4)
                    # 其他情况忽略
    
    # 统计每个值的出现次数
    value_counts = Counter(values)
    
    return values, value_counts

def print_results(values, value_counts):
    """打印统计结果"""
    greater_than_counts = {}
    
    for value in sorted(value_counts.keys()):
        print(f"task {value+1} fail count: {value_counts[value]} 个")
        
        # 计算大于当前值的出现次数
        greater_count = sum(count for val, count in value_counts.items() if val > value)
        greater_than_counts[value] = greater_count
        print(f" task {value+1} success rate: {greater_count/len(values)}")

# 使用示例
if __name__ == "__main__":
    folder_path = "/mnt/workspace/czj/UniWorldModel-0527-t5-COPY/uni_world_model/robot_manipulation_benchmarks/eval_results_threshold0.5skip0epoch20/eval0"  # 替换为实际的文件夹路径
    
    try:
        values, counts = analyze_gif_files(folder_path)
        print_results(values, counts)
    except FileNotFoundError:
        print(f"文件夹 {folder_path} 不存在")
    except Exception as e:
        print(f"发生错误: {e}")