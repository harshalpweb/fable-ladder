"""Smallest possible system: answers nothing. Copy this file and fill in answer()."""
def answer(item):
    # item has: id, bench, kind ('mc' letter, 'num' number, 'text' short answer, 'code' function body), question, and options for 'mc'.
    return "A" if item["kind"] == "mc" else ""
