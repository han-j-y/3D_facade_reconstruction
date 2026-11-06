#!/usr/bin/env python3
"""Measurement only: does SAM3 find the facade on the validation photos? Prompts `facade` and `building`.
For each detection: score, mask share of the image, share of the DSL's windows inside it, and its box against the
window grid's extent. Writes runs_facade_region/regions.json and sheet_<n>.jpg (grid extent green, best facade box
orange, its mask tinted, other detections white)."""
import json
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Sam3Model, Sam3Processor
E = Path(__file__).resolve().parents[1]; O = E / "runs_facade_region"; O.mkdir(exist_ok=True); THR = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Sam3Processor.from_pretrained("facebook/sam3"); sam = Sam3Model.from_pretrained("facebook/sam3").to(device).eval()
Ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
@torch.no_grad()
def detect(im, prompt):
    x = processor(images=im, text=prompt, return_tensors="pt"); x = {k: v.to(device) if hasattr(v, "to") else v for k, v in x.items()}
    x["input_boxes"] = torch.tensor([[[0.5, 0.5, 0.1, 0.1]]], device=device); x["input_boxes_labels"] = torch.tensor([[-10]], device=device)
    r = processor.post_process_instance_segmentation(sam(**x), threshold=THR, mask_threshold=0.5, target_sizes=x["original_sizes"].tolist())[0]
    m = r["masks"]; m = m.detach().float().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    return r["boxes"].detach().float().cpu().numpy(), r["scores"].detach().float().cpu().numpy(), m
out, tiles = {}, []
for i in range(30):
    t = f"v{i:02d}"; D = json.load(open(E / "runs_val30_combined" / t / "facade_dsl.json")); im = Image.open(D["meta"]["image"]).convert("RGB"); iw, ih = im.size
    W = [u["box_xyxy"] for u in D["instances"]]; cx = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in W]
    col, flo = D["meta"]["columns_xy"], D["meta"]["floors_y"]; grid = [col[0][0], flo[0][0], col[-1][1], flo[-1][1]]
    rec = {"size": [iw, ih], "grid": [round(v) for v in grid], "n_elements": len(W)}; best = None
    for prompt in ("facade", "building"):
        boxes, scores, masks = detect(im, prompt); dets = []
        for b, s, m in zip(boxes, scores, masks):
            m = m[0] if m.ndim == 3 else m; mk = m > 0.5
            inbox = sum(b[0] <= x <= b[2] and b[1] <= y <= b[3] for x, y in cx) / max(1, len(cx))
            inmask = sum(mk[min(ih - 1, int(y)), min(iw - 1, int(x))] for x, y in cx) / max(1, len(cx))
            d = dict(score=round(float(s), 3), box=[round(float(v)) for v in b], box_share=round(float((b[2] - b[0]) * (b[3] - b[1]) / (iw * ih)), 3), mask_share=round(float(mk.mean()), 3), in_box=round(float(inbox), 3), in_mask=round(float(inmask), 3))
            dets.append(d)
            if prompt == "facade" and (best is None or (d["in_box"], d["score"]) > (best[0]["in_box"], best[0]["score"])): best = (d, mk)
        rec[prompt] = sorted(dets, key=lambda d: -d["score"])
    out[t] = rec; print(t, D["meta"]["image"].split("/")[-1][:18], rec["size"], "grid", rec["grid"], "| facade", [(d["score"], d["box"], d["in_box"], d["box_share"]) for d in rec["facade"][:3]], "| building", [(d["score"], d["in_box"], d["box_share"]) for d in rec["building"][:2]], flush=True)
    show = im.copy()
    if best:
        tint = np.asarray(show).astype(float); tint[best[1]] = tint[best[1]] * 0.65 + np.array([255, 140, 0]) * 0.35; show = Image.fromarray(tint.astype("uint8"))
    d = ImageDraw.Draw(show); lw = max(2, ih // 250)
    for x in rec["facade"]:
        d.rectangle(x["box"], outline=(255, 255, 255), width=lw)
    if best: d.rectangle(best[0]["box"], outline=(255, 140, 0), width=lw * 2)
    d.rectangle(grid, outline=(40, 220, 60), width=lw * 2)
    H = 430; a = show.resize((round(iw * H / ih), H)); g = ImageDraw.Draw(a)
    lab = f"{t} " + (f"{best[0]['score']:.2f} in {best[0]['in_box']:.2f}" if best else "NONE")
    g.rectangle([0, 0, 8 + 11 * len(lab), 26], fill="black"); g.text((4, 3), lab, fill=(255, 255, 0), font=Ft); tiles.append(a)
json.dump(out, open(O / "regions.json", "w"), indent=1)
W_, x, y, pages, page = 2400, 0, 0, 1, Image.new("RGB", (2400, 1760), "white")
for a in tiles:
    if x + a.width > W_: x = 0; y += 438
    if y + 430 > 1760:
        page.save(O / f"sheet_{pages}.jpg", quality=88); pages += 1; page = Image.new("RGB", (2400, 1760), "white"); x = y = 0
    page.paste(a, (x, y)); x += a.width + 8
page.save(O / f"sheet_{pages}.jpg", quality=88); print("sheets", pages)
