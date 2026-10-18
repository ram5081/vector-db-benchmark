"""
Converts raw benchmark measurements (QPS at a given recall) into real cloud
cost. This is the thesis's actual contribution -- prior work (Kuffo & Boncz,
2025) does this for raw CPUs and libraries, not full self-hosted database
systems; this applies the same principle one layer up the stack.
"""

import math


def cost_per_million_queries(instance_hourly_price_usd: float, measured_qps: float) -> float:
    """Dollars to serve one million queries, given the instance's hourly
    price and the system's measured sustained throughput at some ef_search
    value. This says nothing about recall by itself -- callers should pick
    measured_qps from the ef_search value that meets the target recall
    (see `pick_operating_point` below), not just the fastest one."""
    if measured_qps <= 0:
        return math.inf
    seconds_per_million = 1_000_000 / measured_qps
    hours_per_million = seconds_per_million / 3600
    return instance_hourly_price_usd * hours_per_million


def instances_needed_for_load(target_qps: float, achieved_qps_per_instance: float) -> int:
    """How many instances (of the system under test) are needed to sustain
    a target query load, given one instance's measured throughput."""
    if achieved_qps_per_instance <= 0:
        return math.inf
    return math.ceil(target_qps / achieved_qps_per_instance)


def monthly_cost_at_load(instance_hourly_price_usd: float, target_qps: float,
                          achieved_qps_per_instance: float, hours_per_month: int = 720) -> float:
    """Cost to run enough always-on instances to sustain a target query
    load for a full month -- the number a resource-constrained team
    actually budgets against, as opposed to a per-query cost figure."""
    n = instances_needed_for_load(target_qps, achieved_qps_per_instance)
    if n == math.inf:
        return math.inf
    return n * instance_hourly_price_usd * hours_per_month


def pick_operating_point(sweep_results: list[dict], target_recall: float) -> dict | None:
    """Given a list of {ef_search, recall, qps, p50_latency, ...} dicts for
    ONE system at ONE scale (one row per ef_search value tested), returns
    the first one (lowest ef_search, i.e. cheapest) that meets the target
    recall. Returns None if no ef_search value in the sweep reached it --
    that's a real, reportable finding, not an error to hide."""
    candidates = [r for r in sweep_results if r["recall"] >= target_recall]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["ef_search"])
