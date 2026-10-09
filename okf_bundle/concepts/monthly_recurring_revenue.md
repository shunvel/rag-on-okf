---
type: metric
title: Monthly Recurring Revenue
description: Predictable monthly value of active normalized subscriptions.
resource: bigquery://prod.billing.subscriptions
tags: [finance, mrr, billing, growth]
aliases: [mrr, monthly recurring revenue]
links: []
---

# Monthly Recurring Revenue

Normalize every contract to a monthly amount: annual ÷ 12, quarterly ÷ 3. Always join subscriptions with active billing-adjustment rows so mid-month discounts and credits are reflected.

Exclude one-time professional services unless the board explicitly asks for total revenue. MRR is the denominator for Net Revenue Churn and the numerator for ARPU.
