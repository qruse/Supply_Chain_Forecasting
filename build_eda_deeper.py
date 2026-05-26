from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "EDA.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook():
    cells = []

    cells.append(
        md(
            "# 수요 예측 EDA\n"
            "\n"
            "이 노트북은 `data/DataCoSupplyChainDataset.csv`를 수요 예측과 재고 운영 관점에서 탐색합니다.\n"
            "\n"
            "핵심 질문:\n"
            "- 예측 타깃은 무엇으로 잡는 것이 좋은가?\n"
            "- 상품과 카테고리별 수요는 얼마나 집중되어 있는가?\n"
            "- 수요는 안정적인가, 아니면 간헐적인가?\n"
            "- 뚜렷한 계절성, 요일성, 달력 효과가 있는가?\n"
            "- 예측 시점에 쓸 수 있는 변수와 누수 위험 변수를 어떻게 구분할 것인가?\n"
        )
    )

    cells.append(
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "plt.rcParams.update({\n"
            "    'font.family': 'sans-serif',\n"
            "    'font.sans-serif': ['Noto Sans KR', 'Malgun Gothic', 'AppleSDGothicNeoM00', 'NanumSquare Neo OTF', 'DejaVu Sans'],\n"
            "    'axes.unicode_minus': False,\n"
            "})\n"
            "\n"
            "sns.set_theme(style='whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (12, 4)\n"
            "plt.rcParams['axes.titlesize'] = 13\n"
            "plt.rcParams['axes.labelsize'] = 11\n"
            "\n"
            "DATA_PATH = 'data/DataCoSupplyChainDataset.csv'\n"
            "df = pd.read_csv(DATA_PATH, encoding='latin1', low_memory=False)\n"
            "df['order_dt'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')\n"
            "df['ship_dt'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')\n"
            "df['lead_gap'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']\n"
            "df['order_month'] = df['order_dt'].dt.to_period('M')\n"
            "df['order_day_name'] = df['order_dt'].dt.day_name()\n"
            "df['order_day_num'] = df['order_dt'].dt.day\n"
            "print(df.shape)\n"
        )
    )

    cells.append(
        md(
            "## 1. 데이터 품질과 타깃 선택\n"
            "\n"
            "재고 수요 예측에서 가장 자연스러운 타깃은 `Order Item Quantity`입니다.\n"
            "이 데이터는 거래 단위이므로 같은 상품이 여러 주문에 반복해서 등장합니다. 따라서 예측용 데이터셋을 만들기 전에 일/상품/SKU/카테고리 단위로 집계하는 과정이 필요합니다.\n"
        )
    )

    cells.append(
        code(
            "summary = pd.DataFrame({\n"
            "    'dtype': df.dtypes.astype(str),\n"
            "    'missing_rate': df.isna().mean(),\n"
            "})\n"
            "summary = summary.sort_values('missing_rate', ascending=False)\n"
            "print('주문 날짜 범위:', df['order_dt'].min(), '->', df['order_dt'].max())\n"
            "print('중복 행 수:', df.duplicated().sum())\n"
            "print(summary.head(15).to_string())\n"
        )
    )

    cells.append(
        code(
            "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n"
            "df['Order Item Quantity'].plot(kind='hist', bins=20, ax=ax[0], color='steelblue')\n"
            "ax[0].set_title('Order line quantity distribution')\n"
            "ax[0].set_xlabel('Quantity')\n"
            "\n"
            "df['Order Item Quantity'].plot(kind='box', ax=ax[1])\n"
            "ax[1].set_title('Order line quantity boxplot')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "print(df['Order Item Quantity'].describe().to_string())\n"
        )
    )

    cells.append(
        md(
            "### 해석\n"
            "- 라인 단위 수량은 1~3개에 강하게 몰려 있습니다.\n"
            "- 그래서 원본 거래 행 자체는 예측 단위로는 약하고, 일별 SKU 또는 일별 카테고리 시계열이 훨씬 유용합니다.\n"
            "- 수량이 5로 제한되어 있어 극단값은 크지 않습니다. 따라서 모델링의 큰 난점은 단일 행의 큰 스파이크보다 간헐 수요입니다.\n"
        )
    )

    cells.append(
        md(
            "## 2. 시계열 패턴\n"
            "\n"
            "먼저 전체 시계열을 보고 수요가 안정적인지, 계절성이 있는지, 혹은 데이터 커버리지 문제의 영향을 받는지 확인합니다.\n"
        )
    )

    cells.append(
        code(
            "daily_qty = df.groupby(df['order_dt'].dt.date)['Order Item Quantity'].sum()\n"
            "daily_orders = df.groupby(df['order_dt'].dt.date).size()\n"
            "monthly_qty = df.groupby(df['order_month'])['Order Item Quantity'].sum().sort_index()\n"
            "monthly_orders = df.groupby(df['order_month']).size().sort_index()\n"
            "\n"
            "print('일별 수량 요약')\n"
            "print(daily_qty.describe().to_string())\n"
            "print('일별 주문 건수 요약')\n"
            "print(daily_orders.describe().to_string())\n"
            "\n"
            "fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=False)\n"
            "daily_qty.plot(ax=ax[0], color='steelblue', linewidth=1)\n"
            "ax[0].set_title('Daily demand quantity')\n"
            "ax[0].set_xlabel('Date')\n"
            "ax[0].set_ylabel('Quantity')\n"
            "\n"
            "monthly_qty_ts = monthly_qty.copy()\n"
            "monthly_qty_ts.index = monthly_qty_ts.index.to_timestamp()\n"
            "monthly_qty_ts.plot(ax=ax[1], marker='o', color='darkorange')\n"
            "monthly_qty_ts.rolling(3, min_periods=1).mean().plot(ax=ax[1], color='crimson', linewidth=2, label='3M rolling mean')\n"
            "ax[1].legend()\n"
            "ax[1].set_title('Monthly demand quantity with rolling trend')\n"
            "ax[1].set_xlabel('Month')\n"
            "ax[1].set_ylabel('Quantity')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        code(
            "weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n"
            "weekday_qty = df.groupby('order_day_name')['Order Item Quantity'].sum().reindex(weekday_order)\n"
            "weekday_orders = df.groupby('order_day_name').size().reindex(weekday_order)\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n"
            "weekday_qty.plot(kind='bar', ax=ax[0], color='seagreen')\n"
            "ax[0].set_title('Quantity by weekday')\n"
            "ax[0].set_ylabel('Quantity')\n"
            "\n"
            "weekday_orders.plot(kind='bar', ax=ax[1], color='slateblue')\n"
            "ax[1].set_title('Order count by weekday')\n"
            "ax[1].set_ylabel('Orders')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "\n"
            "day_of_month = df.groupby('order_day_num')['Order Item Quantity'].sum().sort_index()\n"
            "print((day_of_month / day_of_month.sum()).sort_values(ascending=False).head(10).round(4).to_string())\n"
        )
    )

    cells.append(
        md(
            "### 해석\n"
            "- 요일별 수요가 거의 평평해서 주간 계절성은 약합니다.\n"
            "- 달력 일자 효과도 약한 편입니다.\n"
            "- 월별 그래프에서 가장 눈에 띄는 것은 2017-09 이후의 급감인데, 실제 수요 붕괴보다는 부분 관측이나 수집 종료의 영향일 수 있습니다. train/test 분할 시 특히 주의가 필요합니다.\n"
        )
    )

    cells.append(
        md(
            "## 3. 수요 집중도와 ABC 구조\n"
            "\n"
            "수요는 소수의 카테고리와 상품에 강하게 집중되어 있을 가능성이 큽니다. 상위 SKU가 재고 압박의 대부분을 만들기 때문에 재고 정책에서는 매우 중요한 포인트입니다.\n"
        )
    )

    cells.append(
        code(
            "cat_qty = df.groupby('Category Name')['Order Item Quantity'].sum().sort_values(ascending=False)\n"
            "prod_qty = df.groupby('Product Name')['Order Item Quantity'].sum().sort_values(ascending=False)\n"
            "card_qty = df.groupby('Product Card Id')['Order Item Quantity'].sum().sort_values(ascending=False)\n"
            "\n"
            "cat_share = cat_qty / cat_qty.sum()\n"
            "prod_share = prod_qty / prod_qty.sum()\n"
            "card_share = card_qty / card_qty.sum()\n"
            "\n"
            "print('상위 3개 카테고리 비중:', round(cat_share.head(3).sum(), 4))\n"
            "print('상위 5개 카테고리 비중:', round(cat_share.head(5).sum(), 4))\n"
            "print('상위 10개 카테고리 비중:', round(cat_share.head(10).sum(), 4))\n"
            "print('상위 10개 상품 비중:', round(prod_share.head(10).sum(), 4))\n"
            "\n"
            "abc_products = prod_share.cumsum()\n"
            "a_count = (abc_products <= 0.8).sum()\n"
            "b_count = ((abc_products > 0.8) & (abc_products <= 0.95)).sum()\n"
            "c_count = (abc_products > 0.95).sum()\n"
            "print({'A_상품수': int(a_count), 'B_상품수': int(b_count), 'C_상품수': int(c_count)})\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(16, 5))\n"
            "cat_qty.head(10).sort_values().plot(kind='barh', ax=ax[0], color='tomato')\n"
            "ax[0].set_title('Top 10 categories by quantity')\n"
            "ax[0].set_xlabel('Quantity')\n"
            "\n"
            "prod_qty.head(10).sort_values().plot(kind='barh', ax=ax[1], color='goldenrod')\n"
            "ax[1].set_title('Top 10 products by quantity')\n"
            "ax[1].set_xlabel('Quantity')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        code(
            "fig, ax = plt.subplots(1, 2, figsize=(14, 5))\n"
            "pd.Series(card_share.cumsum().values).plot(ax=ax[0], color='purple')\n"
            "ax[0].axhline(0.8, linestyle='--', color='gray')\n"
            "ax[0].set_title('Cumulative demand share by product card')\n"
            "ax[0].set_xlabel('Ranked product card')\n"
            "ax[0].set_ylabel('Cumulative share')\n"
            "\n"
            "pd.Series(prod_share.cumsum().values).plot(ax=ax[1], color='teal')\n"
            "ax[1].axhline(0.8, linestyle='--', color='gray')\n"
            "ax[1].set_title('Cumulative demand share by product name')\n"
            "ax[1].set_xlabel('Ranked product')\n"
            "ax[1].set_ylabel('Cumulative share')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        md(
            "### 해석\n"
            "- 수요 집중도가 매우 높습니다. 소수의 카테고리가 전체 물량 대부분을 설명합니다.\n"
            "- 전형적인 ABC 패턴입니다. 아주 작은 A군이 대부분의 수요를 만들고, long tail은 개별 비중은 작지만 운영상 무시할 수는 없습니다.\n"
            "- 재고 계획에서는 단일 정책보다 A-item과 꼬리 상품을 분리해서 보는 것이 더 적절합니다.\n"
        )
    )

    cells.append(
        md(
            "## 4. 간헐 수요와 변동성\n"
            "\n"
            "상품의 총수요가 높아도 간헐적일 수 있습니다. 수요가 듬성듬성하거나 burst 형태면 단순 예측 모델이 자주 실패하는 지점입니다.\n"
        )
    )

    cells.append(
        code(
            "prod_month = df.groupby(['Product Card Id', 'order_month'])['Order Item Quantity'].sum().unstack(fill_value=0)\n"
            "prod_metrics = pd.DataFrame({\n"
            "    'months_active': (prod_month > 0).sum(axis=1),\n"
            "    'mean_month_qty': prod_month.mean(axis=1),\n"
            "    'std_month_qty': prod_month.std(axis=1),\n"
            "    'cv_month_qty': prod_month.std(axis=1) / prod_month.mean(axis=1).replace(0, np.nan),\n"
            "    'zero_month_share': (prod_month == 0).mean(axis=1),\n"
            "    'total_qty_share': df.groupby('Product Card Id')['Order Item Quantity'].sum() / df['Order Item Quantity'].sum(),\n"
            "})\n"
            "\n"
            "print(prod_metrics[['months_active', 'cv_month_qty', 'zero_month_share', 'total_qty_share']].describe().to_string())\n"
            "\n"
            "stable = prod_metrics[(prod_metrics['months_active'] >= 6) & (prod_metrics['mean_month_qty'] > 20)].copy()\n"
            "intermittent = stable.sort_values(['zero_month_share', 'cv_month_qty'], ascending=False).head(10)\n"
            "volatile = stable.sort_values('cv_month_qty', ascending=False).head(10)\n"
            "print('가장 간헐적인 상품')\n"
            "print(intermittent.to_string())\n"
            "print('가장 변동성이 큰 상품')\n"
            "print(volatile.to_string())\n"
        )
    )

    cells.append(
        code(
            "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n"
            "sns.histplot(prod_metrics['zero_month_share'], bins=20, ax=ax[0], color='steelblue')\n"
            "ax[0].set_title('Distribution of zero-month share by product')\n"
            "ax[0].set_xlabel('Zero month share')\n"
            "\n"
            "sns.histplot(prod_metrics['cv_month_qty'].replace([np.inf, -np.inf], np.nan).dropna(), bins=20, ax=ax[1], color='darkorange')\n"
            "ax[1].set_title('Distribution of monthly CV by product')\n"
            "ax[1].set_xlabel('CV')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        code(
            "cat_month = df.groupby(['Category Name', 'order_month'])['Order Item Quantity'].sum().unstack(fill_value=0)\n"
            "cat_metrics = pd.DataFrame({\n"
            "    'months_active': (cat_month > 0).sum(axis=1),\n"
            "    'mean_month_qty': cat_month.mean(axis=1),\n"
            "    'std_month_qty': cat_month.std(axis=1),\n"
            "    'cv_month_qty': cat_month.std(axis=1) / cat_month.mean(axis=1).replace(0, np.nan),\n"
            "})\n"
            "print(cat_metrics.sort_values('cv_month_qty', ascending=False).head(10).to_string())\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 6))\n"
            "sns.scatterplot(data=prod_metrics, x='mean_month_qty', y='cv_month_qty', alpha=0.7, ax=ax)\n"
            "ax.set_xscale('log')\n"
            "ax.set_title('Product monthly demand: size vs variability')\n"
            "ax.set_xlabel('Mean monthly quantity (log scale)')\n"
            "ax.set_ylabel('Monthly CV')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        md(
            "### 해석\n"
            "- 많은 상품이 전체 관측 월의 일부에만 나타나므로 간헐 수요 문제가 실제로 큽니다.\n"
            "- 물량이 괜찮은 상품도 CV가 높은 경우가 많아서, 시차 변수나 rolling 통계, 계층적 smoothing이 필요합니다.\n"
            "- 그래서 상품 단위 예측은 집계나 추가 메타데이터 없이는 어려운 편입니다.\n"
        )
    )

    cells.append(
        md(
            "## 5. SKU별 무수요 비율\n"
            "\n"
            "모델링 단위가 `SKU x day`라면, SKU마다 수요가 얼마나 자주 0이 되는지 먼저 봐야 합니다.\n"
            "이 값이 높을수록 간헐 수요가 강하고, 너무 높은 SKU는 별도 처리하거나 제외 후보가 될 수 있습니다.\n"
        )
    )

    cells.append(
        code(
            "sku_daily = pd.read_csv('data/sku_daily.csv', encoding='utf-8-sig', parse_dates=['order_date'])\n"
            "sku_daily = sku_daily.sort_values(['sku_id', 'order_date']).reset_index(drop=True)\n"
            "\n"
            "sku_grp = sku_daily.groupby('sku_id')['demand_qty']\n"
            "sku_zero = pd.DataFrame({\n"
            "    'total_days': sku_grp.size(),\n"
            "    'zero_days': sku_grp.apply(lambda s: (s == 0).sum()),\n"
            "    'active_days': sku_grp.apply(lambda s: (s > 0).sum()),\n"
            "    'total_qty': sku_grp.sum(),\n"
            "    'mean_qty': sku_grp.mean(),\n"
            "    'std_qty': sku_grp.std(),\n"
            "})\n"
            "sku_zero['zero_share'] = sku_zero['zero_days'] / sku_zero['total_days']\n"
            "sku_zero['active_share'] = sku_zero['active_days'] / sku_zero['total_days']\n"
            "sku_zero['cv_qty'] = sku_zero['std_qty'] / sku_zero['mean_qty'].replace(0, np.nan)\n"
            "sku_zero = sku_zero.sort_values(['zero_share', 'total_qty'], ascending=[False, False])\n"
            "\n"
            "print(sku_zero[['total_days', 'zero_days', 'active_days', 'zero_share', 'active_share', 'total_qty', 'mean_qty', 'cv_qty']].describe().to_string())\n"
            "print('\\n무수요 비율이 높은 SKU 상위 10개')\n"
            "print(sku_zero.head(10).to_string())\n"
            "\n"
            "thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]\n"
            "print('\\nzero_share 기준 SKU 수')\n"
            "for t in thresholds:\n"
            "    print(f'>= {t:.2f}: {(sku_zero[\"zero_share\"] >= t).sum()}')\n"
        )
    )

    cells.append(
        code(
            "fig, ax = plt.subplots(1, 3, figsize=(18, 5))\n"
            "sns.histplot(sku_zero['zero_share'], bins=20, ax=ax[0], color='steelblue')\n"
            "ax[0].set_title('Distribution of zero-share by SKU')\n"
            "ax[0].set_xlabel('Zero-share')\n"
            "\n"
            "sns.scatterplot(data=sku_zero, x='mean_qty', y='zero_share', ax=ax[1], alpha=0.7)\n"
            "ax[1].set_xscale('log')\n"
            "ax[1].set_title('Mean demand vs zero-share')\n"
            "ax[1].set_xlabel('Mean daily demand (log scale)')\n"
            "ax[1].set_ylabel('Zero-share')\n"
            "\n"
            "sns.scatterplot(data=sku_zero, x='total_qty', y='cv_qty', ax=ax[2], alpha=0.7)\n"
            "ax[2].set_xscale('log')\n"
            "ax[2].set_yscale('log')\n"
            "ax[2].set_title('Total demand vs volatility')\n"
            "ax[2].set_xlabel('Total demand (log scale)')\n"
            "ax[2].set_ylabel('CV (log scale)')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        md(
            "### 해석\n"
            "- SKU별 0 비율이 높으면, 그 품목은 대부분의 날에 팔리지 않는다는 뜻입니다.\n"
            "- 이런 SKU는 개별 예측보다 카테고리 수준으로 묶거나, 아예 별도 정책을 두는 게 더 나을 수 있습니다.\n"
            "- 반대로 0 비율이 낮고 평균 수요가 있는 SKU는 30일 입력 -> 7일 출력 모델의 핵심 타깃으로 남겨둘 가치가 큽니다.\n"
            "- 실무적으로는 `zero_share`와 `total_qty`를 같이 보고, 너무 희소한 SKU를 제외하거나 다른 계층으로 내리는 방식이 유용합니다.\n"
        )
    )

    cells.append(
        md(
            "## 6. 운영 신호와 누수 점검\n"
            "\n"
            "배송 관련 변수는 운영 성과를 진단하는 데는 유용하지만, 사전 수요 예측을 목표로 할 때는 대부분 예측 시점에 사용할 수 없습니다.\n"
        )
    )

    cells.append(
        code(
            "lead_gap = df['lead_gap']\n"
            "print(lead_gap.describe().to_string())\n"
            "print(df.groupby('Shipping Mode')['lead_gap'].agg(['count', 'mean', 'median']).to_string())\n"
            "print(pd.crosstab(df['Shipping Mode'], df['Late_delivery_risk'], normalize='index').round(3).to_string())\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n"
            "sns.boxplot(data=df, x='Shipping Mode', y='lead_gap', ax=ax[0])\n"
            "ax[0].set_title('Lead gap by shipping mode')\n"
            "ax[0].tick_params(axis='x', rotation=20)\n"
            "\n"
            "sns.boxplot(data=df, x='Order Status', y='Order Item Quantity', ax=ax[1])\n"
            "ax[1].set_title('Quantity by order status')\n"
            "ax[1].tick_params(axis='x', rotation=30)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        code(
            "corr_cols = [\n"
            "    'Order Item Quantity',\n"
            "    'Sales',\n"
            "    'Order Item Total',\n"
            "    'Order Item Discount',\n"
            "    'Order Item Discount Rate',\n"
            "    'Order Profit Per Order',\n"
            "    'Order Item Profit Ratio',\n"
            "    'Benefit per order',\n"
            "    'Days for shipping (real)',\n"
            "    'Days for shipment (scheduled)',\n"
            "    'Late_delivery_risk',\n"
            "]\n"
            "corr = df[corr_cols].corr(numeric_only=True)['Order Item Quantity'].sort_values(ascending=False)\n"
            "print(corr.to_frame('수량과의 상관계수').to_string())\n"
        )
    )

    cells.append(
        md(
            "## 7. 예측 시사점\n"
            "\n"
            "모델링 전에 권장하는 설정:\n"
            "- 거래 행을 일별 또는 주별 SKU 단위로 집계하기\n"
            "- A-item과 중요한 카테고리부터 우선적으로 다루기\n"
            "- 랜덤 분할이 아니라 시간 기준 분할 사용하기\n"
            "- 사전 수요 예측에서는 누수 위험이 큰 배송 변수 제외하기\n"
            "- 간헐 수요 대응을 위해 rolling mean, lag, 달력 특성 추가하기\n"
            "\n"
            "결론적으로 이 데이터는 단일 행 예측 모델보다 계층형 수요 예측 파이프라인에 훨씬 잘 맞습니다.\n"
        )
    )

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }
    return nb


def main():
    nb = build_notebook()
    with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
