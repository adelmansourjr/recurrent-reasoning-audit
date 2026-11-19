"""Select authoritative metadata before semantic encoding. No state oracle.

This compiler supports the explicitly documented timestamp/report grammar. It
does not classify the report's meaning or solve dependency readiness.
"""
import re

TIME=re.compile(r"^(?:At time|Time|Status report at|Report timestamp)\s+(\d+)\b")


def newest_report(text):
    candidates=[]
    for line in text.splitlines():
        match=TIME.match(line.strip())
        if match:
            candidates.append((int(match.group(1)),line.strip()))
    if not candidates:
        raise ValueError("No supported timestamped report")
    newest=max(t for t,_ in candidates)
    selected={line for t,line in candidates if t==newest}
    if len(selected)!=1:
        raise ValueError("Conflicting equal-timestamp reports")
    return "Latest evidence only:\n"+next(iter(selected))+"\nClassify status:"
