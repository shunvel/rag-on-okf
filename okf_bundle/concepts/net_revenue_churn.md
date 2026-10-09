---
type: metric
title: Net Revenue Churn
description: Net volume of recurring revenue lost from existing customers after expansion.
resource: bigquery://prod.finance.mrr_table
tags: [finance, nrc, churn, saas]
aliases: [nrc, net revenue churn, net churn]
links:
  - ./monthly_recurring_revenue.md
---

# Net Revenue Churn

Formula: `(Lost MRR - Expansion MRR) / Starting MRR`.

Use this metric for executive reporting. Never substitute **gross logo churn** or raw cancellation counts. Expansion MRR (upsells and price increases) must be subtracted from lost MRR so net churn can be negative when the book is expanding.

Join `mrr_table` on the subscription grain used by Monthly Recurring Revenue. Do not mix finance NRC with support-ticket “churn” anecdotes.
