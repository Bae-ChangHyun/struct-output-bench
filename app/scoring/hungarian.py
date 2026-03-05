"""헝가리안 알고리즘 기반 최대 가중치 이분 매칭.

typo_extraction_benchmark 프로젝트에서 추출.
"""
from __future__ import annotations


def max_weight_matching(scores: list[list[float]]) -> list[tuple[int, int]]:
    """n×m 점수 행렬에서 최적 (i, j) 매칭 쌍을 반환.

    O(k^3) 복잡도 (k = max(n, m)).
    """
    n = len(scores)
    m = len(scores[0]) if n > 0 else 0
    if n == 0 or m == 0:
        return []

    k = max(n, m)
    max_score = max(scores[i][j] for i in range(n) for j in range(m))
    big = max_score + 1.0

    cost = [[big for _ in range(k)] for _ in range(k)]
    for i in range(n):
        for j in range(m):
            cost[i][j] = max_score - scores[i][j]

    u = [0.0] * (k + 1)
    v = [0.0] * (k + 1)
    p = [0] * (k + 1)
    way = [0] * (k + 1)

    for i in range(1, k + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (k + 1)
        used = [False] * (k + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, k + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(0, k + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result: list[tuple[int, int]] = []
    for j in range(1, k + 1):
        i = p[j]
        if i == 0:
            continue
        ii = i - 1
        jj = j - 1
        if ii < n and jj < m:
            result.append((ii, jj))
    return result
