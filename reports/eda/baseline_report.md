# Baseline Analytics Report - Telco Customer Churn

* Customers: **7,043**  |  Overall churn rate: **26.5%**
* Avg tenure: 32.4 months | Avg monthly charge: $64.76

## Key findings
* Month-to-month customers churn at **42.7%** vs **2.8%** on two-year contracts.
* Customers in their first 12 months are the most volatile (see tenure table).
* Tenure is the strongest negative correlate of churn; monthly charges is positively related.

## Churn by contract
| contract | customers | churn_rate_pct | avg_tenure | avg_monthly |
|---|---|---|---|---|
| Month-to-month | 3875 | 42.7 | 18.0 | 66.4 |
| One year | 1473 | 11.3 | 42.0 | 65.0 |
| Two year | 1695 | 2.8 | 56.7 | 60.8 |

## Churn by tenure bucket
| tenure_bucket | churn_rate_pct |
|---|---|
| 0-12m | 47.4 |
| 13-24m | 28.7 |
| 25-48m | 20.4 |
| 49m+ | 9.5 |

## Correlation with churn
| feature | corr_with_churn |
|---|---|
| tenure | -0.352 |
| is_autopay | -0.21 |
| total_charges | -0.198 |
| monthly_charges | 0.193 |
| avg_monthly_spend | 0.193 |
| has_family | -0.163 |
| senior_citizen | 0.151 |
| num_addon_services | -0.088 |

Charts: `churn_distribution.png`, `churn_rate_by_category.png`, `tenure_charges.png`, `correlations.png`.
