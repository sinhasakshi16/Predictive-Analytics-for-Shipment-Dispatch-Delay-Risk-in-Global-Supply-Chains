# Initial DataCo Run Summary

## Dataset audit

- 180,519 order-line rows and 53 columns were read.
- The file contains 65,752 distinct orders and spans January 1, 2015 through January 31, 2018.
- `Late_delivery_risk` is 1 for 98,977 rows (54.83%) and 0 for 81,542 rows.
- The target matches `Delivery Status == "Late delivery"` on every row. This confirms that delivery status is an outcome field and must stay out of prediction features.
- `Product Description` is entirely missing. Four columns contain missing values. The row-hash check found no exact duplicate rows.

## Lightweight temporal benchmark

The latest portion of the data was held out by order date; all line items from an order stayed in the same split. The cutoff was July 9, 2017. The training split has 157,741 rows / 52,601 orders; the test split has 22,778 rows / 13,151 orders, with no overlapping order IDs.

The benchmark averages smoothed training-set late rates for shipping mode, market, region, category, customer segment, order month, and scheduled shipping days. It does not use actual shipping days, delivery status, or shipping date.

| Metric | Result |
|---|---:|
| Accuracy | 0.564 |
| Precision at 0.5 threshold | 0.560 |
| Recall at 0.5 threshold | 0.956 |
| F1 at 0.5 threshold | 0.706 |
| ROC-AUC | 0.732 |
| Average precision (PR-AUC) | 0.796 |

This is an initial ranking benchmark, not evidence of production-ready predictive performance. Its accuracy is only modestly above the 54.8% majority-class rate, and the 0.5 threshold flags most test rows. The next useful step is to evaluate a capacity-limited top-risk policy and compare it with a majority baseline and properly tuned models in an environment where scikit-learn is available.
