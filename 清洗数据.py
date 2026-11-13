import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# =========================================================
# 0. 参数配置
# =========================================================
csv_path = 'Online Retail.csv'
output_rfm_path = 'rfm_result.csv'
output_summary_path = 'rfm_segment_summary.csv'

# 超高金额人工复核阈值
VIP_AMOUNT_THRESHOLD = 50000

db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '123456',      # 修改为你自己的mysql密码
    'database': 'ecommerce_db',
    'charset': 'utf8mb4'
}

# =========================================================
# 1. 读取原始数据
# =========================================================
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(csv_path, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')

# 清理列名BOM、首尾空格
df.columns = (
    df.columns
    .str.replace('\ufeff', '', regex=False)
    .str.strip()
)

print(f"原始数据行数：{df.shape[0]}")
print(f"原始列名：{list(df.columns)}")

# =========================================================
# 2. 数据清洗
# =========================================================
df = df.dropna(subset=['CustomerID'])
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['Total'] = df['Quantity'] * df['UnitPrice']

print(f"清洗后有效交易行数：{df.shape[0]}")

# =========================================================
# 3. 计算 RFM 指标
# =========================================================
base_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

rfm = df.groupby('CustomerID').agg(
    Recency=('InvoiceDate', lambda x: (base_date - x.max()).days),
    Frequency=('InvoiceNo', 'nunique'),
    Monetary=('Total', 'sum')
).reset_index()

print(f"客户总数量：{rfm.shape[0]}")
