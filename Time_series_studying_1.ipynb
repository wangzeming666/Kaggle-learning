#  导入所需的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from pathlib import Path

#  读取数据
data_dir = Path('../input/ts-course-data/')
book_sales = pd.read_csv(
    data_dir / 'book_sales.csv',
    index_col='Date',
    parse_dates=['Date']
)

#  删除我们不使用的 'Paperback' 列
book_sales = book_sales.drop(columns='Paperback')

#  添加时间变量：从 0 到 n-1，用于表示时间趋势
book_sales['Time'] = np.arange(len(book_sales))

#  创建一阶滞后变量：前一天的销售值，用于自回归模型
book_sales['Lag_1'] = book_sales['Hardcover'].shift(1)

#  重新排列列的顺序，便于建模观察
book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])

# ------------------------------------------------------------------------------
#  线性趋势建模（用时间变量预测销量）
# ------------------------------------------------------------------------------

# 移除 NA（开头因为 Lag_1 的 shift 会有空值）
df = book_sales[['Hardcover', 'Time']].dropna()

# 设置特征（时间）和目标（销量）
X = df[['Time']]  # 注意：必须是二维
y = df['Hardcover']

# 拟合线性模型
model = LinearRegression()
model.fit(X, y)

# 预测值，并保留原时间索引
y_pred_trend = pd.Series(model.predict(X), index=df.index)

#  可视化实际销量与拟合趋势线
plt.figure(figsize=(10, 4))
plt.plot(df.index, y, label='实际销量', color='gray')
plt.plot(df.index, y_pred_trend, label='线性趋势预测', color='blue')
plt.title('Hardcover 销量的线性趋势拟合')
plt.xlabel('日期')
plt.ylabel('销量')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
#  自回归建模（AR(1)：用前一天的销量预测今天）
# ------------------------------------------------------------------------------

# 使用滞后变量构建训练集
df_ar = book_sales[['Hardcover']].copy()
df_ar['Lag_1'] = df_ar['Hardcover'].shift(1)

# 设置特征和目标变量，并移除空值
X_ar = df_ar[['Lag_1']].dropna()
y_ar = df_ar['Hardcover']

# 确保 X 和 y 索引完全对齐
y_ar, X_ar = y_ar.align(X_ar, join='inner')

# 拟合 AR(1) 模型
model_ar = LinearRegression()
model_ar.fit(X_ar, y_ar)

# 预测值（与索引对齐）
y_pred_ar = pd.Series(model_ar.predict(X_ar), index=X_ar.index)

#  可视化 AR(1) 拟合效果
plt.figure(figsize=(10, 4))
plt.plot(y_ar.index, y_ar, label='实际销量', color='gray')
plt.plot(y_pred_ar.index, y_pred_ar, label='AR(1) 预测', color='orange')
plt.title('AR(1) 模型对 Hardcover 销量的拟合')
plt.xlabel('日期')
plt.ylabel('销量')
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
#  打印模型系数（用于解释模型规律）
# ------------------------------------------------------------------------------

print(" 线性趋势模型参数:")
print(f"截距 Intercept: {model.intercept_:.2f}")
print(f"斜率 Coefficient: {model.coef_[0]:.4f} （每天销量增加值）")

print("\n AR(1) 模型参数:")
print(f"截距 Intercept: {model_ar.intercept_:.2f}")
print(f"滞后项 Coefficient: {model_ar.coef_[0]:.4f} （前一天销量对今天的影响）")

# ------------------------------------------------------------------------------
#  可选：绘制两个合成 AR 序列（如果你有 ar.csv）
# ------------------------------------------------------------------------------

ar = pd.read_csv(data_dir / 'ar.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar['ar1'], color='teal')
ax1.set_title('Series 1（可能是 AR(1)）')

ax2.plot(ar['ar2'], color='purple')
ax2.set_title('Series 2（可能是 AR(2)）')

plt.tight_layout()
plt.show()
