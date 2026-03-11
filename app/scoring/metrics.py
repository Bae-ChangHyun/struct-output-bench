"""리프 필드 비교 메트릭: NED, 숫자 비교, 부울 비교."""
from __future__ import annotations

from typing import Any


def _levenshtein(a: str, b: str) -> int:
    """O(n*m) DP Levenshtein distance. 외부 의존성 없음."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def ned_similarity(actual: str, predicted: str) -> float:
    """1-NED (Normalized Edit Distance). 1.0 = 완벽 일치, 0.0 = 완전 불일치."""
    if not actual and not predicted:
        return 1.0
    max_len = max(len(actual), len(predicted))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(actual, predicted)
    return 1.0 - (dist / max_len)


def compare_number(actual: Any, predicted: Any, tolerance: float = 0.05) -> float:
    """숫자 비교. exact match = 1.0, ±tolerance 이내 = 1.0, 초과 시 점진적 감소."""
    try:
        a = float(actual)
        p = float(predicted)
    except (TypeError, ValueError):
        return 0.0
    if a == p:
        return 1.0
    if a == 0:
        return 1.0 if abs(p) <= tolerance else 0.0
    rel_error = abs(a - p) / abs(a)
    if rel_error <= tolerance:
        return 1.0
    # tolerance 초과 시 점진적 감소 (1.0에서 0.0으로)
    return max(0.0, 1.0 - (rel_error - tolerance) / (1.0 - tolerance))


def compare_boolean(actual: Any, predicted: Any) -> float:
    """부울 비교. exact match = 1.0."""
    try:
        return 1.0 if bool(actual) == bool(predicted) else 0.0
    except (TypeError, ValueError):
        return 0.0


def compare_leaf(actual: Any, predicted: Any, field_type: str) -> float:
    """타입 기반 리프 필드 비교.

    Args:
        actual: GT 값
        predicted: 추출 값
        field_type: JSON Schema type ("string", "number", "integer", "boolean")

    Returns:
        0.0 ~ 1.0 점수
    """
    if actual is None and predicted is None:
        return 1.0
    if actual is None or predicted is None:
        return 0.0

    if field_type in ("number", "integer"):
        return compare_number(actual, predicted)
    if field_type == "boolean":
        return compare_boolean(actual, predicted)

    # string 또는 fallback: NED
    return ned_similarity(str(actual).strip(), str(predicted).strip())
