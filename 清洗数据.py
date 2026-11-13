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

# =========================================================
# 4. RFM 5分制打分
# =========================================================
rfm['R_score'] = pd.qcut(
    rfm['Recency'].rank(method='first'),
    q=5,
    labels=[5, 4, 3, 2, 1]
).astype(int)

rfm['F_score'] = pd.qcut(
    rfm['Frequency'].rank(method='first'),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm['M_score'] = pd.qcut(
    rfm['Monetary'].rank(method='first'),
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

rfm['RFM_score'] = (
    rfm['R_score'].astype(str)
    + rfm['F_score'].astype(str)
    + rfm['M_score'].astype(str)
)

# =========================================================
# 5. 主分群规则
# =========================================================
def rfm_segment(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']
    if r >= 4 and f >= 4 and m >= 4:
        return '重要价值客户'
    elif r >= 4 and f <= 2 and m >= 4:
        return '重要发展客户'
    elif r <= 2 and f >= 4 and m >= 4:
        return '重要保持客户'
    elif r <= 2 and f <= 2 and m >= 4:
        return '重要挽留客户'
    elif r >= 4 and f >= 4 and m <= 2:
        return '一般价值客户'
    elif r >= 4 and f <= 2 and m <= 2:
        return '一般发展客户'
    elif r <= 2 and f >= 4 and m <= 2:
        return '一般保持客户'
    elif r <= 2 and f <= 2 and m <= 2:
        return '一般挽留客户'
    else:
        return '中间客户/待运营'

rfm['Customer_segment'] = rfm.apply(rfm_segment, axis=1)

# =========================================================
# 6. 中间客户/待运营 二次细分
# =========================================================
def refine_segment(row):
    if row['Customer_segment'] != '中间客户/待运营':
        return row['Customer_segment']
    r, f, m = row['R_score'], row['F_score'], row['M_score']
    if m >= 4:
        if f >= 4:
            return '准重要价值/高价值待激活'
        elif r <= 2:
            return '高额待挽回'
        elif r >= 4:
            return '高潜发展'
        else:
            return '高额中频待观察'
    elif m == 3:
        if f >= 4 and r >= 4:
            return '高频高近中额'
        elif f >= 4:
            return '高频中额待激活'
        elif r >= 4:
            return '高近中额待提升'
        elif r <= 2 and f <= 2:
            return '中额低活跃'
        else:
            return '中频中额待观察'
    else:
        if f >= 4 and r >= 4:
            return '高频高近低客单'
        elif f >= 4:
            return '高频低客单待提升'
        elif r >= 4:
            return '高近低客单待发展'
        else:
            return '低价值待观察'

rfm['Refined_segment'] = rfm.apply(refine_segment, axis=1)

# =========================================================
# 7. 营销策略映射
# =========================================================
strategy_map = {
    '重要价值客户': 'VIP专属权益、积分加倍、优先客服、新品试用、专属折扣',
    '重要发展客户': '提升复购频次、交叉销售、满减券、会员升级引导',
    '重要保持客户': '流失预警、专属召回券、老客回访、定期关怀',
    '重要挽留客户': '高额客户1对1挽回、限时大额券、重点跟踪',
    '一般价值客户': '提升客单价、组合套餐、满赠、关联推荐',
    '一般发展客户': '首复购激励、低门槛券、新客任务、内容种草',
    '一般保持客户': '沉默激活、低成本活动召回、品类推荐',
    '一般挽留客户': '低成本短信/邮件触达、控制投入、沉睡召回',
    '准重要价值/高价值待激活': '准VIP培育、专属券、重点激活、人工跟进',
    '高额待挽回': '高额流失预警、1对1回访、限时大额券、重点跟踪',
    '高潜发展': '提升复购频次、交叉销售、会员升级、专属推荐',
    '高额中频待观察': '高额客户培育、复购激励、定期回访',
    '高频高近中额': '提升客单价、组合套餐、满赠、关联推荐',
    '高频中额待激活': '复购激励、会员权益、品类扩展',
    '高近中额待提升': '提高购买频次、低门槛券、关联推荐',
    '中额低活跃': '沉默激活、活动召回、低成本触达',
    '中频中额待观察': '测试性触达、内容推荐、观察转化',
    '高频高近低客单': '提升客单价、组合套餐、满赠门槛',
    '高频低客单待提升': '提高客单价、关联推荐、满赠',
    '高近低客单待发展': '提升复购频次、低门槛券、会员任务',
    '低价值待观察': '低成本触达、控制投入、观察转化'
}

rfm['Marketing_strategy'] = (
    rfm['Refined_segment']
    .map(strategy_map)
    .fillna('按RFM二次细分后测试性触达')
)
