#!/usr/bin/env python3
# convert_parquet_for_diffsynth_dbg.py
import os, csv, uuid, argparse, sys, traceback
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow.dataset as ds
from PIL import Image, ImageFile

# PIL safety knobs
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def safe_str(x) -> str:
    if x is None:
        return " "
    if isinstance(x, bytes):
        try: x = x.decode("utf-8", "ignore")
        except Exception: return " "
    x = str(x).strip()
    return x if x else " "

def pil_from_any(x) -> Optional[Image.Image]:
    if x is None:
        return None
    try:
        if isinstance(x, Image.Image):
            im = x.convert("RGB"); im.load(); return im
        if isinstance(x, (bytes, bytearray, memoryview)):
            im = Image.open(BytesIO(bytes(x))).convert("RGB"); im.load(); return im
        if isinstance(x, str) and os.path.exists(x):
            with Image.open(x) as f:
                im = f.convert("RGB"); im.load(); return im
        if hasattr(x, "read"):
            im = Image.open(x).convert("RGB"); im.load(); return im
    except Exception:
        return None
    return None

def compose_prompt(rec: Dict[str, object]) -> str:
    instr = safe_str(rec.get("edit_instruction"))
    parts = []
    if instr.strip(): parts.append(instr)
    prompt = " ".join(parts).strip()
    return prompt if prompt else " "

def process_one(inp, out, prompt: str, root: Path) -> Optional[dict]:
    src = pil_from_any(inp)
    tgt = pil_from_any(out)
    if src is None or tgt is None:
        return None
    stem = uuid.uuid4().hex
    ctl_rel = f"control_images/{stem}.png"  # input image
    img_rel = f"images/{stem}.png"          # target/edited image
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "control_images").mkdir(parents=True, exist_ok=True)
    tgt.save(root / img_rel, "PNG")
    src.save(root / ctl_rel, "PNG")
    return {"image": img_rel, "edit_image": ctl_rel, "prompt": prompt}

def main():
    ap = argparse.ArgumentParser("Parquet → DiffSynth folder+CSV (debug friendly).")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 4)))
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--flush_every", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=0, help="max rows; 0 = no limit")
    ap.add_argument("--arrow_threads", type=int, default=0)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)),
                    help="this worker's shard index")
    ap.add_argument("--num_shards", type=int, default=int(os.environ.get("NUM_SHARDS", 1)),
                    help="total number of shards")
    ap.add_argument("--filter_value", default="filtered", help="status value to keep")

    args = ap.parse_args()

    if args.arrow_threads > 0:
        os.environ["ARROW_NUM_THREADS"] = str(args.arrow_threads)

    base = Path(args.out_root)
    base.mkdir(parents=True, exist_ok=True)
    shard_suffix = f".shard{args.shard:03d}-of-{args.num_shards:03d}.csv"
    csv_path = base / ("metadata_edit" + shard_suffix)

    write_header = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fcsv, fieldnames=["image", "edit_image", "prompt"])
    if write_header:
        writer.writeheader()
        fcsv.flush()

    split_path = Path(args.data_root) / args.split
    if not split_path.exists():
        print(f"[ERR] split path not found: {split_path}", file=sys.stderr)
        sys.exit(1)

    arrow = ds.dataset(str(split_path), format="parquet")
    filt = (ds.field("status") == args.filter_value)

    # show how many rows should match BEFORE doing work
    try:
        n_expected = arrow.count_rows(filter=filt)
    except Exception:
        n_expected = None
    print(f"[info] expected rows (status == '{args.filter_value}'): {n_expected}")

    # enumerate fragments and pick only this shard's subset
    frags = list(arrow.get_fragments(filter=filt))
    sel_frags = [f for i, f in enumerate(frags) if i % max(1, args.num_shards) == args.shard]

    written = 0
    seen_batches = 0
    cols = ["input_image", "output_image", "edit_instruction"]

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for frag in sel_frags:
                for batch in frag.to_batches(
                    columns=cols, filter=filt, batch_size=args.batch_size
                ):
                    seen_batches += 1
                    d = batch.to_pydict()
                    inps = d.get("input_image", [])
                    outs = d.get("output_image", [])

                    # Build prompts aligned to rows
                    prompts = []
                    for i in range(len(inps)):
                        rec = {
                            "edit_instruction": d.get("edit_instruction", [None]*len(inps))[i],
                        }
                        prompts.append(compose_prompt(rec))

                    futures = [ex.submit(process_one, inp, out, prm, base)
                               for inp, out, prm in zip(inps, outs, prompts)]

                    batch_ok = 0
                    batch_fail = 0
                    for fut in as_completed(futures):
                        try:
                            row = fut.result()
                        except Exception as e:
                            batch_fail += 1
                            # keep going; print one traceback per batch
                            if batch_fail == 1:
                                print("[warn] worker exception:\n" + "".join(traceback.format_exc()))
                            continue

                        if row is None:
                            batch_fail += 1
                            continue

                        writer.writerow(row)
                        written += 1
                        batch_ok += 1

                        if args.limit and written >= args.limit:
                            print(f"[info] hit limit={args.limit}")
                            fcsv.flush()
                            fcsv.close()
                            print(f"✅ wrote {written} rows total; batches seen={seen_batches}")
                            return

                    fcsv.flush()
                    print(f"[batch {seen_batches}] ok={batch_ok} fail={batch_fail} total_written={written}")

    finally:
        fcsv.close()

    print(f"✅ wrote {written} rows total; batches seen={seen_batches}")

if __name__ == "__main__":
    main()