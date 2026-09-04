"""The `/neurarch train` half of neurarch-bot: one comment starts one real GPU run.

`POST /api/v1/train` is the only endpoint in this product that spends money, so the order
this file holds to is the product and not an implementation detail:

1. **Write access, or nothing.** The key belongs to whoever installed the Action, and a
   comment is something anyone with a GitHub account can leave. Only `OWNER`, `MEMBER` and
   `COLLABORATOR` may spend it.
2. **A key, or nothing.** There is no keyless training and no free tier to fall back to.
3. **The house rules first, then the server.** `.neurarch.yml`'s `policy:` block is the
   repository's own line and `POST /api/v1/train` does not accept one (it plans the graph
   with `buildPlan(graph)` and nothing else), so a policy blocker would sail past the
   server's refusal and be trained. When a policy applies, the bot asks `/api/v1/plan` for
   the verdict first, which costs nothing, and only a legal plan is submitted. The server's
   own 422 stays the backstop for everything structural.
4. **A refusal is the product working.** 422 comes back with the plan that explains it and
   nothing was submitted; the comment says that in those words rather than reading like an
   error. 503 means no GPU worker is configured, and the endpoint deliberately does not fall
   back to a simulation, so neither does this.

The curve is a unicode sparkline and a table, not a picture. GitHub's comment sanitizer
strips `src` from any `<img>` carrying a `data:` URI (markdown and raw HTML alike) and drops
inline `<svg>` outright; this was measured against a real comment on a real repository, not
assumed. A sparkline always renders.

Standard library only, no torch: the argument parsing, the association gate, every refusal
body, the poll loop and the comment upsert are testable against a stub HTTP server.
"""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

TRAIN_PATH = "/api/v1/train"
COMMAND = "/neurarch train"
MARKER = "<!-- neurarch-train -->"
JOB_MARKER = "<!-- neurarch-train:%s -->"
BOT_URL = "https://neurarch.com/bot"
TIMEOUT_SECONDS = 60

DATASETS = ("mnist", "cifar10", "tabular")
OPTIMIZERS = ("adam", "adamw", "sgd")
#: Who may spend the key owner's GPU budget. `author_association` values with write access.
MAY_SPEND = ("OWNER", "MEMBER", "COLLABORATOR")
POLL_SECONDS = 20
DEFAULT_TIMEOUT_MIN = 20
TERMINAL_STATUSES = ("completed", "error", "cancelled", "simulation")
SPARK = "▁▂▃▄▅▆▇█"

#: `key=value` arguments, and what each one turns into on the wire. `model` and `dataset` are
#: read here; the four hyperparameters go into `config`, which the endpoint clamps.
CONFIG_KEYS = {"epochs": "epochs", "batch": "batchSize", "batchsize": "batchSize",
               "lr": "lr", "optimizer": "optimizer"}
KNOWN_KEYS = tuple(sorted(set(list(CONFIG_KEYS) + ["dataset", "gpu", "model"])))


class TrainRequestError(Exception):
    """The endpoint could not be reached; the message is one sentence a reader can act on."""


# ---------------------------------------------------------------- the command

def is_command(body: Optional[str]) -> bool:
    """True when this comment is `/neurarch train`, on its own first line.

    Anchored to the start so quoting someone else's command, or writing about it in a
    sentence, does not book a GPU.
    """
    if not isinstance(body, str):
        return False
    first = body.replace("\r\n", "\n").lstrip().split("\n", 1)[0].strip()
    return first == COMMAND or first.startswith(COMMAND + " ")


def parse_command(body: str) -> Tuple[Dict[str, str], Optional[str]]:
    """`(args, error)` from the command line. An error means nothing is submitted.

    Junk is refused rather than ignored: `epoch=3` (no `s`) silently training the default
    number of epochs is the failure mode this exists to prevent, and a bare word after the
    command is a typo, not a value.
    """
    first = body.replace("\r\n", "\n").lstrip().split("\n", 1)[0].strip()
    rest = first[len(COMMAND):].strip()
    args: Dict[str, str] = {}
    for token in rest.split():
        if "=" not in token:
            return {}, ("`%s` is not `key=value`. Arguments are %s, for example "
                        "`%s epochs=3 dataset=mnist`." % (token, _oxford(KNOWN_KEYS), COMMAND))
        key, value = token.split("=", 1)
        key = key.strip().lower()
        value = value.strip().strip("`")
        if key not in KNOWN_KEYS:
            return {}, ("`%s` is not an argument I know. Arguments are %s." % (key, _oxford(KNOWN_KEYS)))
        if not value:
            return {}, "`%s=` has no value." % key
        if key in args:
            return {}, "`%s` is given twice." % key
        args[key] = value

    if "epochs" in args and _positive_int(args["epochs"]) is None:
        return {}, "`epochs=%s` is not a whole number of epochs." % args["epochs"]
    if "batch" in args and _positive_int(args["batch"]) is None:
        return {}, "`batch=%s` is not a whole number." % args["batch"]
    if "batchsize" in args and _positive_int(args["batchsize"]) is None:
        return {}, "`batchsize=%s` is not a whole number." % args["batchsize"]
    if "lr" in args and _positive_float(args["lr"]) is None:
        return {}, "`lr=%s` is not a positive number." % args["lr"]
    if "dataset" in args and args["dataset"].lower() not in DATASETS:
        return {}, ("`dataset=%s` is not one of %s. The others need a signed upload URL, which "
                    "a comment has no way to obtain." % (args["dataset"], _oxford(DATASETS)))
    if "optimizer" in args and args["optimizer"].lower() not in OPTIMIZERS:
        return {}, "`optimizer=%s` is not one of %s." % (args["optimizer"], _oxford(OPTIMIZERS))
    if "model" in args and ":" not in args["model"]:
        return {}, "`model=%s` needs the form `file.py:ClassOrFactory`." % args["model"]
    return args, None


def _oxford(items: Sequence[str]) -> str:
    items = ["`%s`" % i for i in items]
    if len(items) < 3:
        return " or ".join(items)
    return "%s or %s" % (", ".join(items[:-1]), items[-1])


def _positive_int(text: str) -> Optional[int]:
    try:
        n = int(str(text).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _positive_float(text: str) -> Optional[float]:
    try:
        n = float(str(text).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def may_spend(association: Optional[str]) -> bool:
    return str(association or "").strip().upper() in MAY_SPEND


def config_from_args(args: Dict[str, str]) -> Dict[str, Any]:
    """The four hyperparameters the endpoint accepts, and only the ones that were asked for.

    Never a default: an unasked field left out is a field the server fills in and echoes back,
    and a field this bot invented would be a number nobody chose.
    """
    config: Dict[str, Any] = {}
    for key, wire in CONFIG_KEYS.items():
        if key not in args:
            continue
        if wire == "lr":
            config[wire] = _positive_float(args[key])
        elif wire == "optimizer":
            config[wire] = args[key].lower()
        else:
            config[wire] = _positive_int(args[key])
    return config


def choose_model(models: List[Dict[str, Any]], requested: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """The model to train: the one named, the only one listed, or a question.

    Never a guess when the config lists several: picking the first would train something the
    commenter did not ask for and bill them for it.
    """
    if not models:
        return None, ("`.neurarch.yml` lists no models, so there is nothing to train. Add a "
                      "`models:` entry with `path: file.py:ClassOrFactory` and an `input:`.")
    listed = ", ".join("`%s`" % m["target"] for m in models)
    if requested:
        for m in models:
            if m["target"] == requested or m["target"].endswith(":" + requested) \
                    or m["target"].split(":")[-1] == requested:
                return m, None
        return None, ("`.neurarch.yml` does not list `%s`. It lists %s." % (requested, listed))
    if len(models) == 1:
        return models[0], None
    return None, ("`.neurarch.yml` lists %d models, so say which one: `%s model=%s`. It lists %s."
                  % (len(models), COMMAND, models[0]["target"], listed))


# ---------------------------------------------------------------- the wire

def build_source(target: str, repo: Optional[str], pr: Optional[int],
                 version: str) -> Dict[str, Any]:
    source: Dict[str, Any] = {"kind": "bot", "via": "neurarch-bot/%s" % version, "target": target}
    if repo:
        source["repo"] = repo
    if pr is not None:
        source["pr"] = int(pr)
    return source


def _server_message(body: bytes, fallback: str) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return fallback
    if isinstance(data, dict):
        for key in ("message", "error"):
            if isinstance(data.get(key), str) and data[key]:
                return data[key]
    return fallback


def _json_request(req: urllib.request.Request, api: str,
                  timeout: float) -> Tuple[int, Dict[str, Any], str]:
    """`(status, data, message)`. Raises only when the endpoint could not be reached at all.

    A 422 and a 503 are answers this bot has something to say about, so they come back as
    data rather than as exceptions.
    """
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw, status = resp.read(), resp.getcode()
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (urllib.error.URLError, socket.timeout, ConnectionError, OSError) as e:
        raise TrainRequestError("could not reach %s (%s)" % (api, getattr(e, "reason", None) or e))
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        if status == 200:
            raise TrainRequestError("%s returned a response that is not JSON" % api)
        return status, {}, _server_message(raw, "HTTP %d" % status)
    if not isinstance(data, dict):
        raise TrainRequestError("%s returned a response that is not an object" % api)
    return status, data, _server_message(raw, "HTTP %d" % status)


def submit_train(model: Dict[str, Any], dataset: str, api: str, api_key: str,
                 config: Optional[Dict[str, Any]] = None, gpu: Optional[str] = None,
                 source: Optional[Dict[str, Any]] = None,
                 timeout: float = TIMEOUT_SECONDS) -> Tuple[int, Dict[str, Any], str]:
    """POST the graph. `(status, data, message)`; 200 carries `job_id` and the clamped config."""
    api = api.rstrip("/")
    body: Dict[str, Any] = {"model": model, "dataset": dataset}
    if config:
        body["config"] = config
    if gpu:
        body["gpu"] = gpu
    if source:
        body["source"] = source
    req = urllib.request.Request(
        api + TRAIN_PATH, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "Authorization": "Bearer " + api_key, "User-Agent": "neurarch-bot"})
    return _json_request(req, api, timeout)


def poll_job(job_id: str, api: str, api_key: str,
             timeout: float = TIMEOUT_SECONDS) -> Dict[str, Any]:
    api = api.rstrip("/")
    req = urllib.request.Request(
        "%s%s?job=%s" % (api, TRAIN_PATH, job_id), method="GET",
        headers={"Accept": "application/json", "Authorization": "Bearer " + api_key,
                 "User-Agent": "neurarch-bot"})
    status, data, message = _json_request(req, api, timeout)
    if status != 200:
        raise TrainRequestError("polling %s returned HTTP %d: %s" % (job_id, status, message))
    return data


def is_terminal(job: Dict[str, Any]) -> bool:
    return str(job.get("status") or "") in TERMINAL_STATUSES


def follow_job(job_id: str, api: str, api_key: str, on_change: Callable[[Dict[str, Any]], None],
               timeout_min: float = DEFAULT_TIMEOUT_MIN, poll_seconds: float = POLL_SECONDS,
               sleep: Callable[[float], None] = time.sleep,
               clock: Callable[[], float] = time.monotonic,
               poll: Optional[Callable[[str, str, str], Dict[str, Any]]] = None
               ) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Poll until the job is terminal or the clock runs out. `(last job, timed_out)`.

    `on_change` fires only when the poll returned something new (a new epoch, a new status),
    because the comment is edited from it and an unchanged edit is a wasted API call on a
    loop that runs for twenty minutes.

    A poll that fails is not the end of the run: the network is allowed to blink, so a failed
    poll is logged by the caller through `on_change` never firing and the loop tries again.
    """
    poll = poll or (lambda jid, a, k: poll_job(jid, a, k))
    deadline = clock() + float(timeout_min) * 60.0
    last: Optional[Dict[str, Any]] = None
    seen: Optional[str] = None
    while True:
        try:
            job = poll(job_id, api, api_key)
        except TrainRequestError:
            job = None
        if job is not None:
            last = job
            fingerprint = json.dumps({"status": job.get("status"), "epochs": job.get("epochs"),
                                      "ledger": job.get("ledger")}, sort_keys=True)
            if fingerprint != seen:
                seen = fingerprint
                on_change(job)
            if is_terminal(job):
                return job, False
        if clock() + poll_seconds > deadline:
            return last, True
        sleep(poll_seconds)


# ---------------------------------------------------------------- the numbers

def fmt_money(usd: Optional[float]) -> str:
    if usd is None:
        return "?"
    if usd < 0.01:
        return "$%.4f" % usd
    if usd < 10:
        return "$%.2f" % usd
    return "$%d" % int(round(usd))


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(float(seconds)))
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm %02ds" % (seconds // 60, seconds % 60)
    return "%dh %02dm" % (seconds // 3600, (seconds % 3600) // 60)


def sparkline(values: Sequence[Optional[float]]) -> str:
    """Eight blocks over min..max. A flat series is drawn flat, not as noise."""
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return ""
    low, high = min(nums), max(nums)
    if high - low < 1e-12:
        return SPARK[0] * len(nums)
    span = high - low
    return "".join(SPARK[min(len(SPARK) - 1, int((n - low) / span * (len(SPARK) - 1) + 0.5))]
                   for n in nums)


def epoch_rows(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The per-epoch rows, in order, skipping the utilisation rows the worker interleaves."""
    rows = job.get("epochs")
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("util") is not True
            and r.get("eval") is not True]


def _num(row: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        v = row.get(name)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _cell(value: Optional[float], pct: bool = False) -> str:
    if value is None:
        return ""
    return "%.2f%%" % (100.0 * value) if pct else "%.4f" % value


def render_curve(job: Dict[str, Any]) -> str:
    """The table, with a sparkline above it.

    Not an image: GitHub strips `src` from an `<img>` carrying a `data:` URI, in markdown and
    in raw HTML, and drops inline `<svg>` entirely. Measured, not assumed.
    """
    rows = epoch_rows(job)
    if not rows:
        return ""
    accs = [_num(r, "valAcc", "val_acc", "accuracy") for r in rows]
    losses = [_num(r, "valLoss", "val_loss", "loss") for r in rows]
    out: List[str] = []
    spark_acc = sparkline(accs)
    spark_loss = sparkline(losses)
    if spark_acc:
        out.append("`val acc  %s`  %s" % (spark_acc, _cell(accs[-1], pct=True)))
    if spark_loss:
        out.append("`val loss %s`  %s" % (spark_loss, _cell(losses[-1])))
    if out:
        out.append("")
    out.append("| epoch | train loss | train acc | val loss | val acc |")
    out.append("|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, start=1):
        epoch = r.get("epoch")
        out.append("| %s | %s | %s | %s | %s |" % (
            int(epoch) if isinstance(epoch, (int, float)) else i,
            _cell(_num(r, "trainLoss", "train_loss")),
            _cell(_num(r, "trainAcc", "train_acc"), pct=True),
            _cell(_num(r, "valLoss", "val_loss", "loss")),
            _cell(_num(r, "valAcc", "val_acc", "accuracy"), pct=True)))
    return "\n".join(out)


def final_metric_line(job: Dict[str, Any]) -> str:
    """The one number in bold. `metrics` is the server's own summary of the curve."""
    metrics = job.get("metrics")
    named: Dict[str, float] = {}
    if isinstance(metrics, list):
        for m in metrics:
            if isinstance(m, dict) and isinstance(m.get("name"), str) \
                    and isinstance(m.get("value"), (int, float)):
                named[m["name"]] = float(m["value"])
    parts = []
    if "valAcc" in named:
        parts.append("**val acc %.2f%%**" % (100.0 * named["valAcc"]))
    if "valLoss" in named:
        parts.append("val loss %.4f" % named["valLoss"])
    if parts:
        return " · ".join(parts)
    rows = epoch_rows(job)
    if rows:
        acc = _num(rows[-1], "valAcc", "val_acc", "accuracy")
        if acc is not None:
            return "**val acc %.2f%%**" % (100.0 * acc)
    return ""


def ledger_line(job: Dict[str, Any]) -> str:
    """Whether this run is in the ledger, and therefore whether the next plan will quote it."""
    ledger = job.get("ledger")
    if ledger == "written":
        return ("Written to the outcome ledger: the next `/api/v1/plan` for this structure "
                "will quote this run.")
    if ledger == "already":
        return ("Already in the outcome ledger from an earlier poll; the next "
                "`/api/v1/plan` for this structure quotes this run.")
    if isinstance(ledger, str) and ledger:
        return ("Not written to the outcome ledger (`%s`), so the next plan for this structure "
                "will not quote this run." % ledger)
    return ("No ledger row for this run, so the next plan for this structure will not quote it.")


def trim_error(text: Optional[str], lines: int = 12) -> str:
    if not text:
        return "no output"
    kept = [l for l in str(text).rstrip().splitlines() if l.strip()]
    return "\n".join(kept[-lines:])


# ---------------------------------------------------------------- the comments

def _footer(version: str, host: str, run_url: Optional[str] = None) -> str:
    footer = ("<sub>neurarch-bot v%s. `%s` runs on %s against the API key this repository's "
              "workflow provides; the traced graph was sent to compute the plan, never weights "
              "or data. [How it works](%s)" % (version, COMMAND, host, BOT_URL))
    if run_url:
        footer += " · [run log](%s)" % run_url
    return footer + "</sub>"


def refusal_comment(reason: str, version: str, host: str, run_url: Optional[str] = None) -> str:
    """One short comment, and nothing submitted. The marker carries no job id: there is no job."""
    return "\n".join([MARKER, "## Neurarch training", "", reason, "",
                      _footer(version, host, run_url)]) + "\n"


def access_refusal(actor: Optional[str], association: Optional[str]) -> str:
    who = ("@%s" % actor) if actor else "You"
    assoc = str(association or "NONE").upper()
    return ("**`%s` needs write access to this repository.** It spends the GPU budget of "
            "whoever owns the API key in this repository's secrets, so only an owner, a member "
            "or a collaborator can start a run; %s is `%s`. Nothing was submitted.\n\n"
            "A maintainer can run it for you by leaving the same comment."
            % (COMMAND, who, assoc))


def no_key_refusal() -> str:
    return ("**`%s` needs an API key.** Training is not keyless: the run is booked on a real "
            "GPU and billed to the key owner, so there is nothing to charge it to. Nothing was "
            "submitted.\n\nAdd a Neurarch key (`nrk_...`, Settings → Developer API) as a "
            "repository secret and pass it to the Action:\n\n"
            "```yaml\n- uses: neurarch-ai/neurarch-bot@v0\n  with:\n    mode: train\n"
            "    api-key: ${{ secrets.NEURARCH_API_KEY }}\n```" % COMMAND)


def blocked_comment(target: str, plan: Dict[str, Any], text: str, version: str, host: str,
                    source: str = "server", run_url: Optional[str] = None) -> str:
    """A refused plan. This is the endpoint working, and the comment says so in those words."""
    run = (plan or {}).get("run") or {}
    blockers = [b for b in (run.get("blockers") or []) if isinstance(b, dict)]
    lines = [MARKER, "## Neurarch training: refused", ""]
    summary = (plan or {}).get("summary")
    lines.append("**`%s` will not run, so nothing was submitted and nothing was spent.**" % target)
    lines.append("")
    if summary:
        lines.append("%s" % summary)
        lines.append("")
    if blockers:
        lines.append("**Blockers**")
        for b in blockers:
            detail = str(b.get("detail") or "").strip()
            lines.append("- %s%s" % (b.get("title") or "blocker", (": " + detail) if detail else ""))
            fix = str(b.get("fix") or "").strip()
            if fix:
                lines.append("  - Fix: %s" % fix)
        lines.append("")
    if source == "policy":
        lines.append("This is the `policy:` block in `.neurarch.yml`, this repository's own line. "
                     "`POST /api/v1/train` never sees it, so the bot asks `/api/v1/plan` for the "
                     "verdict before it submits anything: a house rule that only fails after the "
                     "GPU is booked has not saved anyone anything.")
    else:
        lines.append("`POST /api/v1/train` refused this before booking anything. That ordering is "
                     "the point: generating an architecture is cheap and training one that cannot "
                     "forward-pass is not, so the verifier decides whether the money is worth "
                     "spending. Push a fix and comment `%s` again." % COMMAND)
    if text:
        lines += ["", "<details><summary>The plan in full</summary>", "",
                  "```", str(text).rstrip(), "```", "", "</details>"]
    lines += ["", _footer(version, host, run_url)]
    return "\n".join(lines) + "\n"


def gpu_unconfigured_comment(message: str, version: str, host: str,
                             run_url: Optional[str] = None) -> str:
    return refusal_comment(
        "**Managed GPU is not configured for this key, so there is nothing to run on.** "
        "This endpoint does not fall back to a simulation, and neither does this bot: a "
        "simulated curve is not a training run and must not be posted into a pull request as "
        "one. Nothing was submitted.\n\n> %s" % str(message or "").strip(),
        version, host, run_url)


def error_comment(status: int, message: str, version: str, host: str,
                  run_url: Optional[str] = None) -> str:
    return refusal_comment(
        "**`%s` could not submit the run: HTTP %d.** Nothing was submitted.\n\n> %s"
        % (COMMAND, status, str(message or "no message").strip()), version, host, run_url)


def clamp_note(asked: Dict[str, Any], got: Dict[str, Any]) -> str:
    """What the endpoint changed. Silent when it changed nothing."""
    moved = []
    for key in ("epochs", "batchSize", "lr", "optimizer"):
        if key not in asked or asked[key] is None:
            continue
        if key in got and got[key] != asked[key]:
            moved.append("%s %s → %s" % (key, asked[key], got[key]))
    if not moved:
        return ""
    return ("The endpoint clamped what was asked for: %s. Epochs are 1 to 20, batch 8 to 512, "
            "lr 1e-5 to 1, optimizer one of %s." % ("; ".join(moved), _oxford(OPTIMIZERS)))


def run_comment(job_id: str, target: str, dataset: str, config: Dict[str, Any],
                plan: Optional[Dict[str, Any]], version: str, host: str,
                job: Optional[Dict[str, Any]] = None, gpu: Optional[str] = None,
                clamped: str = "", timed_out: bool = False,
                run_url: Optional[str] = None) -> str:
    """The one comment this run owns, from submitted to finished. Keyed on the job id.

    Every update rewrites the whole body rather than appending: a comment that grows a section
    per poll is a log, and the reader wants the run's current state in one screen.
    """
    job = job or {}
    status = str(job.get("status") or "queued")
    rows = epoch_rows(job)
    lines = [JOB_MARKER % job_id, "## Neurarch training run", ""]

    heading = {"completed": ":white_check_mark: finished",
               "error": ":no_entry: the run failed",
               "cancelled": ":no_entry: cancelled",
               "simulation": ":warning: simulated, not a training run"}.get(
        status, ":hourglass_flowing_sand: %s" % status)
    lines.append("**`%s` on `%s`**: %s" % (target, dataset, heading))
    lines.append("")

    asked = ", ".join(filter(None, [
        "%s epochs" % config.get("epochs") if config.get("epochs") is not None else None,
        "batch %s" % config.get("batchSize") if config.get("batchSize") is not None else None,
        "lr %s" % config.get("lr") if config.get("lr") is not None else None,
        str(config.get("optimizer")) if config.get("optimizer") else None,
    ])) or "the endpoint's defaults"
    est = ((plan or {}).get("cost") or {}).get("estCostUsd")
    table = [
        ("Job", "`%s`" % job_id),
        ("Config", asked + (" on %s" % gpu if gpu else "")),
    ]
    if est is not None:
        table.append(("Plan estimate", "%s (roofline, before the run)" % fmt_money(est)))
    if rows and status in TERMINAL_STATUSES:
        table.append(("Epochs", "%d" % len(rows)))
    elif rows:
        table.append(("Progress", "epoch %d%s" % (
            len(rows), " of %s" % config.get("epochs") if config.get("epochs") else "")))
    if job.get("wall_sec"):
        table.append(("Wall clock", fmt_duration(job.get("wall_sec"))))
    if job.get("est_cost_usd") is not None:
        table.append(("Cost", "%s (measured from wall clock at this card's rate)"
                      % fmt_money(job.get("est_cost_usd"))))
    lines.append("| | |")
    lines.append("|---|---|")
    lines += ["| %s | %s |" % (k, v) for k, v in table]
    lines.append("")

    if clamped:
        lines += [clamped, ""]

    curve = render_curve(job)
    if curve:
        lines += [curve, ""]

    if status == "completed":
        final = final_metric_line(job)
        if final:
            lines += [final, ""]
        lines.append(ledger_line(job))
    elif status == "error":
        lines.append("**The run failed, and the check is not red because of it.** This is the "
                     "trainer's error, not a verdict on the pull request: the plan said the graph "
                     "would forward-pass, and something after that did not hold.")
        lines += ["", "```", trim_error(job.get("error_detail")), "```"]
    elif status == "cancelled":
        lines.append("The run was cancelled before it finished.")
    elif status == "simulation":
        lines.append("The worker fell back to a simulation, which is not a training run; there is "
                     "no measurement here and nothing was written to the ledger.")
    elif timed_out:
        lines.append("Still running when this job's `timeout-min` ran out, so this comment stopped "
                     "following it. The run itself is not cancelled: poll it with "
                     "`GET %s?job=%s`." % (TRAIN_PATH, job_id))
    else:
        lines.append("Submitted. This comment is updated as epochs arrive.")

    lines += ["", _footer(version, host, run_url)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- the upsert

def find_comment(comments: Sequence[Dict[str, Any]], marker: str) -> Optional[Dict[str, Any]]:
    """The comment this run owns, by its marker.

    Keyed on the job id, so a training comment is never confused with the plan comment on the
    same pull request, and two runs on one pull request each keep their own.
    """
    for c in comments:
        if isinstance(c, dict) and str(c.get("body") or "").startswith(marker):
            return c
    return None
