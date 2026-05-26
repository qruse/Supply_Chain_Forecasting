from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "DataCoSupplyChainDataset.csv"
NOTEBOOK_PATH = ROOT / "EDA.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook():
    cells = []
    cells.append(
        md(
            "# 수요 예측 관점 EDA\n"
            "\n"
            "이 노트북은 `DataCoSupplyChainDataset.csv`를 기준으로 재고 수요 예측에 도움이 되는 탐색을 수행합니다.\n"
            "\n"
            "중점 포인트:\n"
            "- 수요 타깃 후보 정의: 주문 라인 수량(`Order Item Quantity`)과 주문 건수\n"
            "- 시계열 패턴: 일별, 월별, 요일별 변동\n"
            "- 수요 집중도: 카테고리/상품별 long-tail 여부\n"
            "- 운영 변수: 배송 리드타임, 지연 위험, 할인\n"
            "- 주의사항: 주문 후 알 수 있는 변수는 예측 시점에서 누수 가능성이 있음\n"
        )
    )

    cells.append(
        code(
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "\n"
            "sns.set_theme(style='whitegrid')\n"
            "plt.rcParams['figure.figsize'] = (12, 4)\n"
            "plt.rcParams['axes.titlesize'] = 13\n"
            "plt.rcParams['axes.labelsize'] = 11\n"
            "\n"
            "DATA_PATH = r'data/DataCoSupplyChainDataset.csv'\n"
            "df = pd.read_csv(DATA_PATH, encoding='latin1', low_memory=False)\n"
            "df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')\n"
            "df['shipping date (DateOrders)'] = pd.to_datetime(df['shipping date (DateOrders)'], errors='coerce')\n"
            "df['lead_gap'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']\n"
            "df['order_day'] = df['order date (DateOrders)'].dt.day_name()\n"
            "df['order_month'] = df['order date (DateOrders)'].dt.to_period('M').astype(str)\n"
            "print('shape:', df.shape)\n"
        )
    )

    cells.append(
        md(
            "## 1. 데이터 개요\n"
            "\n"
            "수요 예측에서는 보통 주문이 발생한 시점 기준으로 미래 수요를 예측합니다.\n"
            "이 데이터는 주문 라인 단위라서, 기본 타깃은 `Order Item Quantity`가 적절합니다.\n"
            "대안으로는 일자/상품/카테고리 단위로 집계한 주문 건수도 사용할 수 있습니다.\n"
        )
    )

    cells.append(
        code(
            "print('shape:', df.shape)\n"
            "print('order date range:', df['order date (DateOrders)'].min(), '->', df['order date (DateOrders)'].max())\n"
            "print('shipping date range:', df['shipping date (DateOrders)'].min(), '->', df['shipping date (DateOrders)'].max())\n"
            "\n"
            "missing = df.isna().mean().sort_values(ascending=False)\n"
            "print(missing.head(15).to_frame('missing_rate').to_string())\n"
            "print('duplicate rows:', df.duplicated().sum())\n"
        )
    )

    cells.append(
        md(
            "### 해석 포인트\n"
            "- `Product Description`은 전부 결측이고, `Order Zipcode`도 결측 비율이 매우 높습니다.\n"
            "- 수요 예측 모델 입력에서는 이런 컬럼을 그대로 쓰기보다 제외하거나 별도 처리하는 편이 낫습니다.\n"
            "- 주문/배송 관련 컬럼 중 일부는 주문 시점 이후 정보라서, 예측 시점에 사용할 수 있는지 먼저 확인해야 합니다.\n"
        )
    )

    cells.append(
        code(
            "print('late delivery risk vs delivery status')\n"
            "print(pd.crosstab(df['Delivery Status'], df['Late_delivery_risk'], normalize='index').round(3).to_string())\n"
            "\n"
            "print('order status share')\n"
            "print(df['Order Status'].value_counts(normalize=True).round(4).to_frame('share').to_string())\n"
            "\n"
            "print('shipping mode share')\n"
            "print(df['Shipping Mode'].value_counts(normalize=True).round(4).to_frame('share').to_string())\n"
        )
    )

    cells.append(
        md(
            "## 2. 수요 시계열 패턴\n"
            "\n"
            "재고 수요 예측에서는 시계열 집계가 핵심입니다.\n"
            "여기서는 일별 수량과 월별 수량을 확인해서 추세와 변동성을 봅니다.\n"
        )
    )

    cells.append(
        code(
            "daily_qty = df.groupby(df['order date (DateOrders)'].dt.date)['Order Item Quantity'].sum()\n"
            "daily_orders = df.groupby(df['order date (DateOrders)'].dt.date).size()\n"
            "monthly_qty = df.groupby(df['order date (DateOrders)'].dt.to_period('M'))['Order Item Quantity'].sum()\n"
            "monthly_orders = df.groupby(df['order date (DateOrders)'].dt.to_period('M')).size()\n"
            "\n"
            "print(daily_qty.describe().to_frame('daily_quantity').to_string())\n"
            "print(daily_orders.describe().to_frame('daily_order_count').to_string())\n"
            "\n"
            "fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=False)\n"
            "daily_qty.plot(ax=ax[0], color='steelblue', linewidth=1)\n"
            "ax[0].set_title('Daily demand volume (quantity)')\n"
            "ax[0].set_xlabel('Date')\n"
            "ax[0].set_ylabel('Quantity')\n"
            "\n"
            "monthly_qty.index = monthly_qty.index.to_timestamp()\n"
            "monthly_qty.plot(ax=ax[1], marker='o', color='darkorange')\n"
            "ax[1].set_title('Monthly demand volume (quantity)')\n"
            "ax[1].set_xlabel('Month')\n"
            "ax[1].set_ylabel('Quantity')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        code(
            "weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']\n"
            "weekday_qty = df.groupby(df['order date (DateOrders)'].dt.day_name())['Order Item Quantity'].sum().reindex(weekday_order)\n"
            "weekday_orders = df.groupby(df['order date (DateOrders)'].dt.day_name()).size().reindex(weekday_order)\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n"
            "weekday_qty.plot(kind='bar', ax=ax[0], color='seagreen')\n"
            "ax[0].set_title('Quantity by weekday')\n"
            "ax[0].set_xlabel('')\n"
            "ax[0].set_ylabel('Quantity')\n"
            "\n"
            "weekday_orders.plot(kind='bar', ax=ax[1], color='slateblue')\n"
            "ax[1].set_title('Order count by weekday')\n"
            "ax[1].set_xlabel('')\n"
            "ax[1].set_ylabel('Orders')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        md(
            "### 해석 포인트\n"
            "- 요일별 패턴이 거의 균일하면 주간 계절성은 약한 편입니다.\n"
            "- 월별 패턴은 2017년 10월 이후 급감하는 구간이 보여서, 데이터 수집 종료/부분 월 여부를 확인해야 합니다.\n"
            "- 시계열 분할 시에는 이 급감 구간을 테스트 셋으로 그대로 쓰기보다, 데이터 커버리지를 검토하는 것이 좋습니다.\n"
        )
    )

    cells.append(
        code(
            "cat_qty = df.groupby('Category Name')['Order Item Quantity'].sum().sort_values(ascending=False)\n"
            "prod_qty = df.groupby('Product Name')['Order Item Quantity'].sum().sort_values(ascending=False)\n"
            "\n"
            "top_cat_share = cat_qty.head(10).sum() / cat_qty.sum()\n"
            "top_prod_share = prod_qty.head(10).sum() / prod_qty.sum()\n"
            "print(f'top 3 category share: {cat_qty.head(3).sum() / cat_qty.sum():.4f}')\n"
            "print(f'top 5 category share: {cat_qty.head(5).sum() / cat_qty.sum():.4f}')\n"
            "print(f'top 10 category share: {top_cat_share:.4f}')\n"
            "print(f'top 10 product share: {top_prod_share:.4f}')\n"
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
        md(
            "### 해석 포인트\n"
            "- 수요가 상위 몇 개 카테고리/상품에 매우 집중되어 있습니다.\n"
            "- 이 데이터는 long-tail 성격이 강하므로, 전체 공통 모델 1개보다 상위 상품군별 모델이나 계층형 접근이 유리할 수 있습니다.\n"
            "- 재고 운영에서는 상위 SKU에 더 촘촘한 안전재고 정책을 두는 것이 효과적입니다.\n"
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
            "print(corr.to_frame('corr_with_quantity').to_string())\n"
            "\n"
            "print('lead_gap summary')\n"
            "print(df['lead_gap'].describe().to_frame('lead_gap').to_string())\n"
            "\n"
            "fig, ax = plt.subplots(1, 2, figsize=(14, 4))\n"
            "sns.boxplot(data=df, x='Shipping Mode', y='lead_gap', ax=ax[0])\n"
            "ax[0].set_title('Lead gap by shipping mode')\n"
            "ax[0].set_xlabel('Shipping Mode')\n"
            "ax[0].set_ylabel('Actual - scheduled days')\n"
            "ax[0].tick_params(axis='x', rotation=20)\n"
            "\n"
            "discount_bins = pd.cut(df['Order Item Discount Rate'], bins=[-0.001, 0, 0.05, 0.1, 0.2, 1], include_lowest=True)\n"
            "discount_means = df.groupby(discount_bins)['Order Item Quantity'].mean()\n"
            "discount_means.plot(kind='bar', ax=ax[1], color='mediumpurple')\n"
            "ax[1].set_title('Average quantity by discount-rate bin')\n"
            "ax[1].set_xlabel('Discount rate bin')\n"
            "ax[1].set_ylabel('Average quantity')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
        )
    )

    cells.append(
        md(
            "## 3. 수요 예측 관점 최종 해석\n"
            "\n"
            "1. **타깃은 `Order Item Quantity`가 가장 자연스럽다.** 주문 라인 단위의 실수요를 가장 직접적으로 반영한다.\n"
            "2. **수요는 특정 카테고리와 상품에 강하게 집중되어 있다.** 상위 3개 카테고리가 전체 수량의 절반 이상을 차지한다.\n"
            "3. **요일 효과는 약하고, 월별 급변 구간은 데이터 커버리지 이슈 가능성이 있다.**\n"
            "4. **배송 관련 변수는 예측 시점 누수 위험이 있다.** 주문 시점에 사용할 수 있는 변수와 후행 변수를 분리해야 한다.\n"
            "5. **할인과 수량의 단순 상관은 거의 없다.** 할인 효과를 보려면 단순 상관보다 프로모션/카테고리별 상호작용을 봐야 한다.\n"
            "\n"
            "### 모델링 전 추천\n"
            "- 일별/주별로 상품 또는 카테고리 단위 집계\n"
            "- 시간 기반 train/validation split\n"
            "- 상위 SKU와 나머지 SKU를 분리한 계층형 모델\n"
            "- 주문 시점 이후 정보는 feature에서 제외\n"
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
