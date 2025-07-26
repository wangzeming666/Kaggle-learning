{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 📘 时间序列线性建模与自回归分析（含中文注释）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📦 导入所需的库\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from pathlib import Path"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📁 读取数据\n",
    "data_dir = Path('../input/ts-course-data/')\n",
    "book_sales = pd.read_csv(\n",
    "    data_dir / 'book_sales.csv',\n",
    "    index_col='Date',\n",
    "    parse_dates=['Date']\n",
    ")\n",
    "\n",
    "# 🧹 删除我们不使用的 'Paperback' 列\n",
    "book_sales = book_sales.drop(columns='Paperback')\n",
    "\n",
    "# 🕒 添加时间变量：从 0 到 n-1，用于表示时间趋势\n",
    "book_sales['Time'] = np.arange(len(book_sales))\n",
    "\n",
    "# 🔁 创建一阶滞后变量：前一天的销售值，用于自回归模型\n",
    "book_sales['Lag_1'] = book_sales['Hardcover'].shift(1)\n",
    "\n",
    "# 📊 重新排列列的顺序，便于建模观察\n",
    "book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📈 线性趋势建模（用时间变量预测销量）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 移除 NA（开头因为 Lag_1 的 shift 会有空值）\n",
    "df = book_sales[['Hardcover', 'Time']].dropna()\n",
    "\n",
    "# 设置特征（时间）和目标（销量）\n",
    "X = df[['Time']]\n",
    "y = df['Hardcover']\n",
    "\n",
    "# 拟合线性模型\n",
    "model = LinearRegression()\n",
    "model.fit(X, y)\n",
    "\n",
    "# 预测值，并保留原时间索引\n",
    "y_pred_trend = pd.Series(model.predict(X), index=df.index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📊 可视化实际销量与拟合趋势线\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(df.index, y, label='实际销量', color='gray')\n",
    "plt.plot(df.index, y_pred_trend, label='线性趋势预测', color='blue')\n",
    "plt.title('Hardcover 销量的线性趋势拟合')\n",
    "plt.xlabel('日期')\n",
    "plt.ylabel('销量')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔁 自回归建模（AR(1)：用前一天的销量预测今天）"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 使用滞后变量构建训练集\n",
    "df_ar = book_sales[['Hardcover']].copy()\n",
    "df_ar['Lag_1'] = df_ar['Hardcover'].shift(1)\n",
    "\n",
    "# 设置特征和目标变量，并移除空值\n",
    "X_ar = df_ar[['Lag_1']].dropna()\n",
    "y_ar = df_ar['Hardcover']\n",
    "\n",
    "# 确保 X 和 y 索引完全对齐\n",
    "y_ar, X_ar = y_ar.align(X_ar, join='inner')\n",
    "\n",
    "# 拟合 AR(1) 模型\n",
    "model_ar = LinearRegression()\n",
    "model_ar.fit(X_ar, y_ar)\n",
    "\n",
    "# 预测值（与索引对齐）\n",
    "y_pred_ar = pd.Series(model_ar.predict(X_ar), index=X_ar.index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📊 可视化 AR(1) 拟合效果\n",
    "plt.figure(figsize=(10, 4))\n",
    "plt.plot(y_ar.index, y_ar, label='实际销量', color='gray')\n",
    "plt.plot(y_pred_ar.index, y_pred_ar, label='AR(1) 预测', color='orange')\n",
    "plt.title('AR(1) 模型对 Hardcover 销量的拟合')\n",
    "plt.xlabel('日期')\n",
    "plt.ylabel('销量')\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 📌 打印模型系数（用于解释模型规律）\n",
    "print(\"📈 线性趋势模型参数:\")\n",
    "print(f\"截距 Intercept: {model.intercept_:.2f}\")\n",
    "print(f\"斜率 Coefficient: {model.coef_[0]:.4f} （每天销量增加值）\")\n",
    "\n",
    "print(\"\\n🔁 AR(1) 模型参数:\")\n",
    "print(f\"截距 Intercept: {model_ar.intercept_:.2f}\")\n",
    "print(f\"滞后项 Coefficient: {model_ar.coef_[0]:.4f} （前一天销量对今天的影响）\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
