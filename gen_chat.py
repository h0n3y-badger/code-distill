"""Generate general-chat / instruction data from the 32B teacher.

No execution gate exists for open chat, so we use a quality gate: valid message
list, a substantive assistant reply, roles alternating user/assistant, and no
canned refusal. The 32B general teacher is the point here — this is the data
that gives the coding student a real conversational range. Appends to CHAT_RAW;
ctrl-C safe; stops at TARGET_CHAT.
"""
import json
import config_uni as C
from genlib import ask, extract_json, count_lines
from datalib import is_valid_messages

SYS = ("You create high-quality, natural assistant training conversations. "
       "Return ONLY one JSON object, no prose, no markdown fences.")

_REFUSAL = ("i cannot", "i can't", "i'm unable", "as an ai language model",
            "i am unable", "i'm sorry, but i can", "i am not able")


def build_prompt(topic, style):
    return (
        f"Create ONE realistic {style} on the topic of {topic}. "
        f"Write it as a natural exchange between a curious human user and a "
        f"helpful, knowledgeable assistant. The assistant's replies should be "
        f"accurate, well-structured, and genuinely useful — the quality you'd "
        f"want a model to learn. Vary length to fit the question. "
        f'Respond with JSON: {{"messages": [{{"role": "user", "content": "..."}}, '
        f'{{"role": "assistant", "content": "..."}} ...]}}  '
        f"(2 to 6 messages, always starting with user and alternating.)")


def good(msgs):
    if not is_valid_messages(msgs):
        return False
    if msgs[0].get("role") != "user":
        return False
    roles = [m.get("role") for m in msgs]
    if any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)):
        return False  # must alternate
    asst = [m["content"] for m in msgs if m.get("role") == "assistant"]
    if not asst or min(len(a) for a in asst) < 40:
        return False  # substantive replies only
    if any(a.strip().lower().startswith(_REFUSAL) for a in asst):
        return False
    return True


def main():
    kept = count_lines(C.CHAT_RAW)
    seen = rejected = 0
    print(f"chat: resuming at {kept}/{C.TARGET_CHAT} kept")
    combos = [(t, s) for t in C.CHAT_TOPICS for s in C.CHAT_STYLES]
    with open(C.CHAT_RAW, "a") as out:
        # loop the grid until target; temperature keeps repeats diverse
        while kept < C.TARGET_CHAT:
            for topic, style in combos:
                if kept >= C.TARGET_CHAT:
                    break
                seen += 1
                try:
                    obj = extract_json(ask(build_prompt(topic, style), SYS))
                    msgs = obj["messages"]
                except Exception:
                    continue
                if not good(msgs):
                    rejected += 1
                    continue
                # normalize to just role/content
                msgs = [{"role": m["role"], "content": m["content"].strip()}
                        for m in msgs]
                out.write(json.dumps({
                    "messages": msgs, "kind": "chat",
                    "topic": topic, "style": style}) + "\n")
                out.flush(); kept += 1
                if kept % 25 == 0:
                    print(f"chat: kept={kept} seen={seen} rejected={rejected}",
                          flush=True)
    print(f"chat DONE. kept={kept} seen={seen} rejected={rejected} -> {C.CHAT_RAW}")


if __name__ == "__main__":
    main()
