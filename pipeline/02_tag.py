"""Stage 3-4: stratified sample -> Claude Haiku relevance + codebook tagging (one call per post).
Resumable: re-running skips ids already in data/tagged.jsonl.

  python3 pipeline/02_tag.py --plan          # show sample sizes + cost estimate, no API calls
  python3 pipeline/02_tag.py --limit 50      # smoke test on 50 posts
  python3 pipeline/02_tag.py                 # full run
"""
import argparse, asyncio, json, os, sys, time
import pandas as pd
import anthropic
from codebook import SCHEMA, SYSTEM, render_post

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(DATA, "tagged.jsonl")
MODEL = "claude-haiku-5-5"
MAX_CHARS = 6000  # very long Reddit posts are cut here; flagged with truncated=True

# posts per source (highest rule-priority first; Play/App Store weighted to Groww)
QUOTA = {"Reddit": 6000, "YouTube": 2500, "Play Store": 2500, "App Store": 548, "Forums": 1200, "Social": 55}

def build_sample(df):
    parts = []
    for src, n in QUOTA.items():
        s = df[df.source == src].copy()
        if src in ("Play Store", "App Store"):
            s["is_groww"] = (s.group == "Groww").astype(int)
            s = s.sort_values(["is_groww", "priority"], ascending=False)
        else:
            s = s.sort_values("priority", ascending=False)
        parts.append(s.head(n))
    return pd.concat(parts)

async def tag_one(client, sem, r, fh, stats):
    text = r["text"]
    truncated = len(text) > MAX_CHARS
    r = dict(r, text=text[:MAX_CHARS])
    async with sem:
        for attempt in range(4):
            try:
                resp = await client.messages.create(
                    model=MODEL,
                    max_tokens=2000,
                    system=SYSTEM,
                    output_config={"effort": "low",
                                   "format": {"type": "json_schema", "schema": SCHEMA}},
                    messages=[{"role": "user", "content": render_post(r)}],
                )
                if resp.stop_reason == "refusal":
                    stats["refusal"] += 1
                    return
                out = json.loads(next(b.text for b in resp.content if b.type == "text"))
                out.update(id=r["id"], truncated=truncated, model=MODEL)
                stats["in"] += resp.usage.input_tokens
                stats["out"] += resp.usage.output_tokens
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                stats["ok"] += 1
                return
            except anthropic.RateLimitError:
                await asyncio.sleep(10 * (attempt + 1))
            except (anthropic.APIStatusError, anthropic.APIConnectionError, json.JSONDecodeError, StopIteration) as e:
                if attempt == 3:
                    stats["err"] += 1
                    print(f"  failed {r['id']}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                await asyncio.sleep(2 * (attempt + 1))

async def main(args):
    df = pd.read_parquet(os.path.join(DATA, "corpus_filtered.parquet"))
    sample = build_sample(df)
    sample.to_parquet(os.path.join(DATA, "sample.parquet"), index=False)
    print("sample per source:", sample.groupby("source").size().to_dict(), "total", len(sample))
    est_in = (sample.text.str.len().clip(upper=MAX_CHARS) / 3.5 + 1600).sum()
    est_out = len(sample) * 450
    print(f"estimated tokens: in {est_in/1e6:.1f}M, out {est_out/1e6:.1f}M -> ~${est_in/1e6*0.10 + est_out/1e6*0.50:.2f} on Haiku 5.5")
    if args.plan:
        return
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    todo = [r for r in sample.to_dict("records") if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"already tagged {len(done)}, tagging {len(todo)} now")
    client = anthropic.AsyncAnthropic(max_retries=3)
    sem = asyncio.Semaphore(args.concurrency)
    stats = dict(ok=0, err=0, refusal=0, **{"in": 0, "out": 0})
    t0 = time.time()
    with open(OUT, "a") as fh:
        tasks = [tag_one(client, sem, r, fh, stats) for r in todo]
        for i, t in enumerate(asyncio.as_completed(tasks), 1):
            await t
            if i % 250 == 0 or i == len(tasks):
                cost = stats["in"] / 1e6 * 0.10 + stats["out"] / 1e6 * 0.50
                print(f"  {i}/{len(tasks)}  ok={stats['ok']} err={stats['err']} refusal={stats['refusal']}  "
                      f"${cost:.2f}  {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    asyncio.run(main(ap.parse_args()))
