---
type: catalog
title: SaaS Finance OKF Bundle
description: Canonical metric nodes for the Enterprise Knowledge Gateway demo.
resource: okf://saas-finance
tags: [okf, saas, finance]
links:
  - ./concepts/net_revenue_churn.md
  - ./concepts/monthly_recurring_revenue.md
  - ./concepts/customer_acquisition_cost.md
  - ./concepts/average_revenue_per_user.md
---

# SaaS Finance Knowledge Catalog

These nodes are the **authoritative** definitions. Warehouse snapshots and chat logs may disagree with them. Retrieval should prefer this catalog once a metric is identified.

| Node | Aliases | Depends on |
| --- | --- | --- |
| Net Revenue Churn | nrc, churn | Monthly Recurring Revenue |
| Monthly Recurring Revenue | mrr | — |
| Customer Acquisition Cost | cac | Monthly Recurring Revenue |
| Average Revenue Per User | arpu | Monthly Recurring Revenue |
