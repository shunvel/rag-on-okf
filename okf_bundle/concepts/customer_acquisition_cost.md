---
type: metric
title: Customer Acquisition Cost
description: Fully loaded sales and marketing cost to acquire one net-new customer.
resource: postgres://prod.marketing.attribution_ledger
tags: [marketing, cac, attribution, cost]
aliases: [cac, customer acquisition cost]
links:
  - ./monthly_recurring_revenue.md
---

# Customer Acquisition Cost

Formula: `Total Sales & Marketing Spend / Number of New Customers Acquired`.

Exclude organic and direct sign-ups from the denominator. Include fully loaded S&M (salaries, tools, paid media). Pair CAC with MRR when computing payback: `CAC / (ARPU × Gross Margin)`.
