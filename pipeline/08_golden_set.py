"""Golden set: pick 20 AI-tagged posts, export a blind labelling sheet (xlsx with dropdowns),
and keep the AI answers in a separate file for scoring with 09_score_golden.py."""
import os, sys, json
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from codebook import SCHEMA
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUT = os.path.join(D, "golden")
os.makedirs(OUT, exist_ok=True)

t = pd.read_parquet(os.path.join(D, "tags_ai.parquet"))
s = pd.read_parquet(os.path.join(D, "tag_sample.parquet"))[["id", "source", "group", "thread_title", "title", "text", "rating"]]
df = t.merge(s, on="id")
df = df[df.text.str.len().between(80, 1100)]

# stratified pick: (source, condition, n)
PLAN = [
    ("Reddit", lambda d: d.is_relevant & d.continuation_signal.isin(["at_risk", "dropped"]), 4),
    ("Reddit", lambda d: d.is_relevant & (d.continuation_signal == "likely_continue"), 2),
    ("Reddit", lambda d: ~d.is_relevant, 2),
    ("Play Store", lambda d: d.is_relevant & (d.group == "Groww"), 3),
    ("Play Store", lambda d: ~d.is_relevant, 1),
    ("App Store", lambda d: d.is_relevant, 2),
    ("YouTube", lambda d: d.is_relevant, 2),
    ("YouTube", lambda d: ~d.is_relevant, 1),
    ("Forums", lambda d: d.is_relevant, 1),
    ("Social", lambda d: d.is_relevant, 1),
    ("Reddit", lambda d: d.is_relevant & (d.confidence == "low"), 1),  # deliberately ambiguous
]
picked = []
for src, cond, n in PLAN:
    pool = df[(df.source == src) & cond(df) & ~df.id.isin([p.id for p in picked])]
    picked += list(pool.sample(min(n, len(pool)), random_state=11).itertuples())
G = pd.DataFrame(picked).drop(columns="Index").sample(frac=1, random_state=3).reset_index(drop=True)
G.insert(0, "item", range(1, len(G) + 1))

FIELDS = ["is_relevant", "journey_stage", "sip_action", "triggers", "continuation_signal",
          "emotion", "outside_app_help", "income_stability", "influenced_by"]
MULTI = {"triggers", "outside_app_help"}
P = SCHEMA["properties"]
def allowed(f):
    if f == "is_relevant": return ["TRUE", "FALSE"]
    spec = P[f]; return spec["items"]["enum"] if f in MULTI else spec["enum"]

G[["item", "id"] + FIELDS + ["unmet_need", "evidence_quote", "confidence"]].to_json(
    os.path.join(OUT, "golden_ai_answers.jsonl"), orient="records", lines=True, force_ascii=False)

wb = Workbook(); ws = wb.active; ws.title = "Label here"
head = ["item", "id", "source / app", "context (thread or review title)", "post text"] + FIELDS + ["your note (optional)"]
ws.append(head)
for c in ws[1]:
    c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="1F4E79")
    c.alignment = Alignment(wrap_text=True, vertical="top")
lists = wb.create_sheet("allowed values")
for j, f in enumerate(FIELDS, 1):
    vals = allowed(f)
    lists.cell(1, j, f).font = Font(bold=True)
    for i, v in enumerate(vals, 2): lists.cell(i, j, v)
    col = ws.cell(1, 5 + j).column_letter
    if f not in MULTI:
        dv = DataValidation(type="list", formula1=f"='allowed values'!${lists.cell(1, j).column_letter}$2:${lists.cell(1, j).column_letter}${len(vals)+1}", allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"{col}2:{col}{len(G)+1}")
for r in G.itertuples():
    ctx = " | ".join(x for x in [r.thread_title or "", r.title or "", f"rating {r.rating}" if r.rating else ""] if x)
    ws.append([r.item, r.id, f"{r.source} / {r.group}", ctx, r.text] + [""] * len(FIELDS) + [""])
widths = [5, 13, 18, 30, 80] + [18] * len(FIELDS) + [30]
for i, w in enumerate(widths, 1): ws.column_dimensions[ws.cell(1, i).column_letter].width = w
for row in ws.iter_rows(min_row=2):
    for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
ws.freeze_panes = "F2"
guide = wb.create_sheet("how to label", 0)
for line in [
    "Golden set: label 20 posts yourself BEFORE seeing the AI's answers.",
    "",
    "1. Go to the 'Label here' tab. Read each post (and its context column).",
    "2. Fill the 9 coloured columns using the dropdowns. Rules are the same the AI used:",
    "   - is_relevant: TRUE only if the post says something about a person's own (or a specific other person's) SIP/MF experience, behaviour, worry or in-app experience.",
    "     Generic advice, market news, trading/IPO, plain praise ('good app') = FALSE. If FALSE, you can leave the other columns blank.",
    "   - journey_stage: how long the writer has been doing SIPs (unknown if not stated).",
    "   - sip_action: what they did. If several, pick the most severe: redeemed > stopped > paused > reduced > skipped > considering_stopping > switched > continued.",
    "   - triggers and outside_app_help can have several values: type them separated by commas, copied from the 'allowed values' tab (or 'none' / 'none_mentioned').",
    "   - continuation_signal: likely_continue / at_risk (doubting, confused, considering stopping) / dropped (paused, stopped, redeemed) / unclear.",
    "   - income_stability, influenced_by: only if stated, otherwise unknown.",
    "3. Do not guess demographics. Use the note column for anything ambiguous.",
    "4. Save the file (keep the name) and tell Claude. Scoring: python3 pipeline/09_score_golden.py",
]:
    guide.append([line])
guide.column_dimensions["A"].width = 160
path = os.path.join(OUT, "golden_set_LABEL_ME.xlsx")
wb.save(path)
print(G.groupby(["source", "is_relevant"]).size().to_string())
print("wrote", path)
