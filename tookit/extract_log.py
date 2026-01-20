import re
import pandas as pd
import os
import argparse
import openpyxl

def parse():
    parser = argparse.ArgumentParser(description="解析日志文件并保存为 Excel")
    parser.add_argument("--input_log", type=str, default="/mnt/innovator/data/wangcong/data/eval/eval_Qwen3-30B-A3B-Base_20260115_080237.log", help="输入的日志文件路径")
    parser.add_argument("--output_excel", type=str, default="/mnt/innovator/data/wangcong/data/eval/results/general_task.xlsx", help="输出的 Excel 文件名")
    parser.add_argument("--description", type=str, help="description for this run", default="None")
    args = parser.parse_args()
    
    return args


def parse_main_metrics(file_path):
    """
    只解析顶级大项：过滤掉空首列、以 '-' 开头的子项、以及带缩进的子项
    """
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return None, None

    # 自动从文件名提取模型名
    base_name = os.path.basename(file_path)
    model_name = base_name.replace("eval_", "").split("_202")[0].replace(".log", "")

    results = {}
    
    # 正则提取：Group/Task, Metric, Value
    table_line_re = re.compile(r'^\|(?P<task>[^|]+)\|[^|]*\|[^|]*\|[^|]*\|(?P<metric>[^|]*)\|[^|]*\|(?P<value>[^|]+)\|')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 1. 基本过滤：必须是表格行，且不是表头
            if not line.startswith("|") or "Metric" in line or "---" in line:
                continue
            
            match = table_line_re.match(line)
            if match:
                # 获取原始的第一列内容（保留空格以便判断缩进）
                raw_task_col = match.group('task')
                
                # --- 核心过滤逻辑 ---
                # A. 检查是否为空白列（子指标行）
                if raw_task_col.isspace():
                    continue
                
                # B. 检查是否以 '-' 开头（lm-eval 常见的子任务标识）
                # C. 检查是否以空格开头（有些格式下子任务通过缩进标识）
                stripped_task = raw_task_col.strip()
                
                # 如果原始列以空格开头，或者 strip 后以 '-' 开头，说明是子项
                if raw_task_col.startswith(" ") or stripped_task.startswith("-"):
                    continue
                
                # D. 排除表头关键词
                if stripped_task.lower() in ["tasks", "groups"]:
                    continue
                # ------------------

                value = match.group('value').strip()
                
                # 每一项只记录第一次出现的那个分值（通常是主指标）
                if stripped_task not in results:
                    try:
                        results[stripped_task] = float(value)
                    except ValueError:
                        results[stripped_task] = value

    return model_name, results

def update_excel(log_path, excel_path):
    """
    将解析结果追加到 Excel 中
    """
    # 1. 解析当前的 log 文件
    model_name, new_data = parse_main_metrics(log_path)
    if not new_data:
        print("解析失败，未提取到有效得分。")
        return

    # 2. 读取现有 Excel 或创建新的
    if os.path.exists(excel_path):
        print(f"检测到现有表格 {excel_path}，正在追加列...")
        df_existing = pd.read_excel(excel_path, index_col=0)
    else:
        print(f"未检测到表格，将创建新表 {excel_path}...")
        df_existing = pd.DataFrame()

    # 3. 将新数据转换为 Series，名称设为模型名
    new_series = pd.Series(new_data, name=model_name)
    new_series = pd.to_numeric(new_series, errors='ignore') * 100

    # 4. 合并数据
    # 如果模型名（列名）已存在，则会覆盖旧列
    # 如果数据集（行名）不存在，则会自动新增一行
    # if model_name in df_existing.columns:
    #     print(f"警告：模型 {model_name} 已存在于表中，将更新其数据。")
    #     df_existing[model_name] = new_series
    # else:
    #     df_existing = pd.concat([df_existing, new_series], axis=1)
    df_existing = pd.concat([df_existing, new_series], axis=1)

    # 5. 保存结果
    # 按照索引名称（数据集名称）排序，让表格更整齐
    df_existing.sort_index(inplace=True)
    df_existing.to_excel(excel_path)
    
    print("-" * 30)
    print(f"更新成功！")
    print(f"当前模型 (列): {model_name}")
    print(f"数据集 (行): 共有 {len(df_existing)} 个数据集已录入")
    print(f"表格路径: {os.path.abspath(excel_path)}")
    print("-" * 30)

def extra_info(df, info=["SUM","AVG","Description"]):
    pass

def main():
    # --- 配置区域 ---
    # 1. 输入的 log 文件路径
    args = parse()
    input_log = args.input_log
    output_excel = args.output_excel
    # ----------------

    model_name, data = parse_main_metrics(input_log)
    
    update_excel(input_log, output_excel)

if __name__ == "__main__":
    main()