import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError 
# ======================================================== 
# 0. 参数配置 
# ========================================================
csv_path = 'Online Retail.csv'
output_rfm_path = 'rfm_result.csv'
output_summary_path = 'rfm_segment_summary.csv' 
# 超高金额人工复核阈值 
VIP_AMOUNT_THRESHOLD = 50000 
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root', 
    'password': '123456', # 修改为你自己的 mysql 密码
    'database': 'ecommerce_db',
    'charset': 'utf8mb4' 
}
# ======================================================== 
# 1. 读取原始数据 
# ========================================================
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(csv_path, encoding='gbk')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')
