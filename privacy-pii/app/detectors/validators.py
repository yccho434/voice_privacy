# app/detectors/validators.py
from __future__ import annotations
import re

# ---------- Luhn(신용카드) ----------
def luhn_ok(num: str) -> bool:
    digits = [int(c) for c in re.sub(r"\D", "", num)]
    if len(digits) < 12:  # 너무 짧으면 거절
        return False
    s = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9: d -= 9
        s += d
    return s % 10 == 0

# ---------- 주민등록번호(체크섬) ----------
# 형식: YYMMDD-ABCDEFG (13자리)
def rrn_ok(rrn: str) -> bool:
    s = re.sub(r"\D", "", rrn)
    if len(s) != 13: return False
    weights = [2,3,4,5,6,7,8,9,2,3,4,5]
    total = sum(int(a)*b for a,b in zip(s[:12], weights))
    check = (11 - (total % 11)) % 10
    return check == int(s[12])

# ---------- 사업자등록번호(체크섬) ----------
# 형식: 3-2-5 → 총 10자리
def biz_ok(biz: str) -> bool:
    s = re.sub(r"\D", "", biz)
    if len(s) != 10: return False
    weights = [1,3,7,1,3,7,1,3,5]
    total = sum(int(s[i]) * weights[i] for i in range(9))
    total += (int(s[8]) * 5) // 10
    check = (10 - (total % 10)) % 10
    return check == int(s[9])

# ---------- 이메일 도메인 허용/차단 ----------
def email_domain_allowed(email: str,
                         allow: set[str] | None = None,
                         deny: set[str] | None = None) -> bool:
    m = re.search(r"@([A-Za-z0-9.-]+)$", email)
    if not m: return False
    dom = m.group(1).lower()
    if deny and any(dom.endswith(d) for d in deny):  # 차단 우선
        return False
    if allow:
        return any(dom.endswith(a) for a in allow)
    return True
