"""리프 필드 비교 메트릭: NED, 숫자 비교, 부울 비교."""
from __future__ import annotations

import unicodedata
from typing import Any

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "0", "no", "n", "f", ""}


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


def _norm_ned(value: Any) -> str:
    """NED(관대) 비교용 정규화: 유니코드 NFC + 양끝 공백 제거 + 대소문자 무시."""
    return unicodedata.normalize("NFC", str(value)).strip().casefold()


def _norm_exact(value: Any) -> str:
    """엄격 비교용 정규화: 유니코드 NFC + 양끝 공백 제거 (대소문자 구분)."""
    return unicodedata.normalize("NFC", str(value)).strip()


def _is_numeric(value: Any) -> bool:
    """실제로 숫자로 해석 가능한 값인지. union(int|str)에서 'DQ' 같은 문자열을 걸러낸다."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
            return True
        except ValueError:
            return False
    return False


def _to_bool(value: Any) -> bool | None:
    """값을 bool로 해석. 해석 불가면 None (bool() 강제변환의 'false'→True 방향 오류를 피함)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().casefold()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
    return None


def ned_similarity(actual: str, predicted: str) -> float:
    """1-NED (Normalized Edit Distance). 1.0 = 완벽 일치, 0.0 = 완전 불일치."""
    if not actual and not predicted:
        return 1.0
    max_len = max(len(actual), len(predicted))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(actual, predicted)
    return 1.0 - (dist / max_len)


def compare_number(actual: Any, predicted: Any, tolerance: float = 0.05, integer: bool = False) -> float:
    """숫자 비교. integer=True면 정확 일치만(연도·식별자). 아니면 ±tolerance 이내 1.0, 초과 시 점진 감소."""
    try:
        a = float(actual)
        p = float(predicted)
    except (TypeError, ValueError):
        return 0.0
    if a == p:
        return 1.0
    if integer:
        # 정수/식별자(연도 등)는 근사 허용이 오히려 오답을 만점 처리하므로 정확 일치만 인정.
        return 0.0
    if a == 0:
        return 1.0 if abs(p) <= tolerance else 0.0
    rel_error = abs(a - p) / abs(a)
    if rel_error <= tolerance:
        return 1.0
    return max(0.0, 1.0 - (rel_error - tolerance) / (1.0 - tolerance))


def compare_boolean(actual: Any, predicted: Any) -> float:
    """부울 비교. 문자열 'false'/'0'도 올바르게 False로 해석. 해석 불가면 0.0."""
    a = _to_bool(actual)
    p = _to_bool(predicted)
    if a is None or p is None:
        return 0.0
    return 1.0 if a == p else 0.0


def compare_leaf(actual: Any, predicted: Any, field_type: str) -> float:
    """타입 기반 리프 필드 비교 (NED 기반, 부분점수 허용).

    Args:
        field_type: JSON Schema type ("string", "number", "integer", "boolean")
    """
    if actual is None and predicted is None:
        return 1.0
    if actual is None or predicted is None:
        return 0.0

    if field_type in ("number", "integer"):
        # union(int|str) 등으로 실제 값이 숫자가 아니면(예: "DQ", "N/A") 문자열 비교로 폴백.
        if _is_numeric(actual) and _is_numeric(predicted):
            return compare_number(actual, predicted, integer=(field_type == "integer"))
        return ned_similarity(_norm_ned(actual), _norm_ned(predicted))
    if field_type == "boolean":
        return compare_boolean(actual, predicted)

    return ned_similarity(_norm_ned(actual), _norm_ned(predicted))


def compare_leaf_exact(actual: Any, predicted: Any, field_type: str) -> float:
    """엄격 일치 메트릭. NED의 관대함을 보완하는 0/1 지표.

    문자열은 NFC 정규화 후 대소문자까지 완전 일치해야 1.0, 숫자는 허용오차 없이 동일해야 1.0.
    """
    if actual is None and predicted is None:
        return 1.0
    if actual is None or predicted is None:
        return 0.0

    if field_type in ("number", "integer"):
        if _is_numeric(actual) and _is_numeric(predicted):
            return 1.0 if float(actual) == float(predicted) else 0.0
        return 1.0 if _norm_exact(actual) == _norm_exact(predicted) else 0.0
    if field_type == "boolean":
        return compare_boolean(actual, predicted)

    return 1.0 if _norm_exact(actual) == _norm_exact(predicted) else 0.0
