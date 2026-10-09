---
type: metric
title: Average Revenue Per User
description: Average monthly recurring revenue contributed by each paying customer.
resource: bigquery://prod.finance.mrr_table
tags: [finance, arpu, unit-economics, saas]
aliases: [arpu, average revenue per user, average revenue per customer]
links:
  - ./monthly_recurring_revenue.md
---

# Average Revenue Per User

Formula: `MRR / Paying Customers`.

Use paying customers only. Seat-weight multi-tier accounts. Do not divide booked GAAP revenue by total registered users — that understates ARPU and breaks LTV:CAC.
