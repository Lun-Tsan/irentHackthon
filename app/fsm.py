"""The two state machines from the system diagrams (圖 3 車輛, 圖 4 車損紀錄).

Every transition is explicit; anything not listed raises InvalidTransition,
so the demo can never drift into a state the diagrams don't have.
"""


class InvalidTransition(Exception):
    pass


VEHICLE = {
    "states": {
        "available": "可租",
        "rented": "出租中",
        "pending_review": "待確認",
        "needs_service": "待清潔／待維修",
        "pending_acceptance": "待驗收",
    },
    "transitions": {
        ("available", "pickup"): "rented",
        ("available", "flag"): "needs_service",            # 取車回報嚴重髒汙／損傷，或人工確認需維修
        ("rented", "return"): "pending_review",
        ("pending_review", "clear"): "available",          # 無異常／純外觀小傷／普通
        ("pending_review", "flag"): "needs_service",       # 髒汙／安全件／需維修
        ("needs_service", "service_done"): "pending_acceptance",
        ("pending_acceptance", "accept"): "available",
        ("pending_acceptance", "reject"): "needs_service",
    },
}

DAMAGE = {
    "states": {
        "candidate": "候選",
        "confirmed": "已確認（歸本次租約）",
        "gap": "空窗期（歸公司／保險）",
        "closed": "關閉（誤報／舊傷）",
        "appealing": "申訴中",
        "repaired": "已修復",
    },
    "transitions": {
        ("candidate", "confirm"): "confirmed",
        ("candidate", "gap"): "gap",
        ("candidate", "dismiss"): "closed",
        ("confirmed", "appeal"): "appealing",
        ("appealing", "uphold"): "confirmed",
        ("appealing", "overturn"): "closed",
        ("confirmed", "repair"): "repaired",
        ("gap", "repair"): "repaired",
    },
}

WORKORDER_STATES = {"open": "新建", "in_progress": "處理中", "pending_acceptance": "待驗收", "done": "結案", "rejected": "驗收不通過"}


def step(machine: dict, state: str, event: str) -> str:
    key = (state, event)
    if key not in machine["transitions"]:
        raise InvalidTransition(f"{state} --{event}--> (沒有這條邊)")
    return machine["transitions"][key]


def label(machine: dict, state: str) -> str:
    return machine["states"].get(state, state)
