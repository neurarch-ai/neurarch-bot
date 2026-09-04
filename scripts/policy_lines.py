"""Where a policy rule lives: the line of `.neurarch.yml` that a policy finding came from.

A policy blocker reads `policy: max_params 500K exceeded (1.1M)`, and a reader who thinks the
rule is wrong has nowhere to go: the comment names the verdict, not the decision behind it. A
rule is a decision somebody on the team made, and a comment that cannot point at it makes the
rule read as Neurarch's opinion. So the bot re-reads the config it sent, finds the line each
violated rule was written on, and prints it under the findings.

Line numbers cannot come from `yaml.safe_load` (a parsed mapping has no lines), so this is a
small indentation-aware scan of the same text. It only ever looks for the seven policy keys,
which means the worst a shape it does not understand can do is fail to find a line: the plan
itself is not touched, and an unlocated rule prints as unlocated rather than as a wrong line.
"""
import re
from typing import Any, Dict, List, Optional

# The keys the plan endpoint enforces. `max_params`, `must_fit`, `max_layers`, `forbid_types`
# and `require_types` block; `max_memory_gb` and `max_cost_usd` warn.
POLICY_KEYS = (
    "max_params", "must_fit", "max_layers", "forbid_types", "require_types",
    "max_memory_gb", "max_cost_usd",
)

TOP_BLOCK = "the top-level `policy:` block"
MODEL_BLOCK = "this model's own `policy:` block"

_UNREADABLE = re.compile(r"unreadable line \(([a-z_]+)\s*:")
_UNKNOWN_IN = re.compile(r"\bin ([a-z_]+) \(")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def rule_text(line: str) -> str:
    """The rule as written, without its trailing comment.

    The comment is often a paragraph explaining the decision, and wrapping it into a bullet
    reads as noise; whoever follows the link sees it in place, which is where it belongs.
    """
    out: List[str] = []
    depth, quote = 0, None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "#" and depth == 0 and (i == 0 or line[i - 1].isspace()):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _key_of(body: str) -> str:
    return body.split(":", 1)[0].strip()


def locate_rules(text: str) -> Dict[str, Any]:
    """Every policy key in a `.neurarch.yml`, with the 1-based line it is written on.

    Returns `{"top": {key: {"line": n, "text": "max_params: 200M  # comment"}},
              "models": {"models/cnn.py:SmallCNN": {key: {...}}}}`.

    A model's own block is keyed by its `path:`, which is the same string `parse_models` uses
    as the target, so a finding for one model can be resolved without re-parsing the config.
    """
    top: Dict[str, Dict[str, Any]] = {}
    models: Dict[str, Dict[str, Dict[str, Any]]] = {}
    section: Optional[str] = None
    list_indent: Optional[int] = None
    item: Optional[Dict[str, Any]] = None

    def flush() -> None:
        if item and item["path"] and item["rules"]:
            models.setdefault(item["path"], {}).update(item["rules"])

    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = _indent(raw)
        body = stripped

        if indent == 0 and not body.startswith("- "):
            flush()
            item = None
            list_indent = None
            section = _key_of(body)
            continue

        if section == "policy":
            key = _key_of(body)
            if key in POLICY_KEYS:
                top[key] = {"line": number, "text": rule_text(stripped)}
            continue

        if section != "models":
            continue

        if body.startswith("- ") and (list_indent is None or indent == list_indent):
            flush()
            list_indent = indent
            item = {"path": None, "rules": {}, "policy_indent": None}
            body = body[2:].strip()
            indent += 2
        if item is None:
            continue

        key = _key_of(body)
        if item["policy_indent"] is not None and indent <= item["policy_indent"]:
            item["policy_indent"] = None  # the nested block ended; this key is the model's own
        if key == "path":
            item["path"] = body.split(":", 1)[1].split("#")[0].strip().strip("\"'")
        elif key == "policy":
            item["policy_indent"] = indent
        elif item["policy_indent"] is not None and key in POLICY_KEYS:
            item["rules"][key] = {"line": number, "text": rule_text(stripped)}
    flush()
    return {"top": top, "models": models}


def rule_key(finding: Dict[str, Any]) -> Optional[str]:
    """The policy key one finding came from, or None when the finding is not a policy one.

    The endpoint says it three ways: `detail` carries `max_params: 500K` for a rule that was
    read and broken, and for the two rules it could not read at all the key is only in the
    title (`unreadable line (max_params: ...)`, `unknown GPU in must_fit (banana)`). A rule
    the server could not read is exactly the one a reader most needs pointed at.
    """
    if not isinstance(finding, dict):
        return None
    detail = str(finding.get("detail") or "")
    key = detail.split(":", 1)[0].strip()
    if key in POLICY_KEYS:
        return key
    title = str(finding.get("title") or "")
    if not title.startswith("policy:") and finding.get("stage") != "policy":
        return None
    for pattern in (_UNREADABLE, _UNKNOWN_IN):
        match = pattern.search(title)
        if match and match.group(1) in POLICY_KEYS:
            return match.group(1)
    rest = title.split("policy:", 1)[-1].strip()
    word = rest.split(" ", 1)[0].strip()
    return word if word in POLICY_KEYS else None


def violated_keys(plan: Dict[str, Any]) -> List[str]:
    """The policy keys this plan's blockers and warnings fired on, in the order they appear."""
    run = (plan or {}).get("run") if isinstance(plan, dict) else None
    if not isinstance(run, dict):
        return []
    keys: List[str] = []
    for finding in list(run.get("blockers") or []) + list(run.get("warnings") or []):
        key = rule_key(finding)
        if key and key not in keys:
            keys.append(key)
    return keys


def _reference(config_name: str, line: Optional[int], blob_url: Optional[str]) -> str:
    if line is None:
        return "`%s`" % config_name
    label = "`%s:%d`" % (config_name, line)
    if blob_url:
        return "[%s](%s/%s#L%d)" % (label, blob_url.rstrip("/"), config_name, line)
    return label


def rule_lines(plan: Dict[str, Any], locations: Dict[str, Any], target: str,
               config_name: str = ".neurarch.yml", blob_url: Optional[str] = None) -> List[str]:
    """One bullet per violated rule: the line as written, and where it is written.

    A model's own block wins over the top-level one, because that is the order the two are
    merged in before the request goes out; quoting the top-level line for a rule the model
    overrode would point at a line that had no part in the verdict.
    """
    per_model = (locations.get("models") or {}).get(target) or {}
    top = locations.get("top") or {}
    out: List[str] = []
    for key in violated_keys(plan):
        found = per_model.get(key)
        where = MODEL_BLOCK
        if found is None:
            found = top.get(key)
            where = TOP_BLOCK
        if found is None:
            out.append("- `%s` fired, and no line of %s spells it out: it reached the plan from "
                       "somewhere else (a config this run did not read)."
                       % (key, "`%s`" % config_name))
            continue
        out.append("- `%s` is at %s, in %s." % (found["text"], _reference(config_name, found["line"], blob_url), where))
    return out


def render_section(lines: List[str]) -> str:
    """The section that goes under the findings. Empty when no policy rule fired."""
    if not lines:
        return ""
    parts = ["**The rules that fired, and where they live**", ""]
    parts.extend(lines)
    parts.append("")
    parts.append("_Each of these is a decision someone here made, not a Neurarch default. "
                 "If a rule is wrong, that line is the thing to change._")
    return "\n".join(parts)
