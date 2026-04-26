#!/usr/bin/env python3
"""
Inpainting Demo Server
Runs the C++ inpaint binary and serves results via Flask.
"""

import os
import subprocess
import time
import math
import json
from flask import Flask, request, jsonify, send_file, send_from_directory
from PIL import Image
import io
import base64
import numpy as np

app = Flask(__name__, static_folder='.', template_folder='.')

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGES_DIR   = os.environ.get("IMAGES_DIR",  "../images")
INPAINT_BIN  = os.environ.get("INPAINT_BIN", "../build/inpaint")
UPLOAD_DIR   = os.path.join(IMAGES_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_metrics(img_a: np.ndarray, img_b: np.ndarray) -> dict:
    """Compute pixel-level quality metrics between two RGB images."""
    diff = np.abs(img_a.astype(np.int32) - img_b.astype(np.int32))
    per_pixel = diff.sum(axis=2)          # sum over channels
    total     = per_pixel.size

    differing = int((per_pixel > 0).sum())
    avg_diff  = float(per_pixel.mean())
    max_diff  = int(per_pixel.max())
    stddev    = float(per_pixel.std())

    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = 100.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)

    quality = "GOOD" if psnr > 35 else ("ACCEPTABLE" if psnr > 25 else "POOR")

    return {
        "total_pixels":     total,
        "differing_pixels": differing,
        "differing_pct":    round(100.0 * differing / total, 4),
        "avg_diff":         round(avg_diff, 4),
        "max_diff":         max_diff,
        "stddev":           round(stddev, 4),
        "psnr":             round(psnr, 4),
        "quality":          quality,
    }


def mat_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_inpaint(image_path: str, mode: str) -> tuple[float, str]:
    """
    Run the inpaint binary.
    mode: 'cpu' (1 thread) or 'gpu'
    Returns (elapsed_seconds, output_image_path).
    """
    ##
    base   = os.path.splitext(image_path)[0]
    suffix = "GPU" if mode == "gpu" else "CPU"
    out    = f"{base}.bmp_output_{suffix}.png"
    cpu_out = f"{base}.bmp_output_CPU.png"
    gpu_out = f"{base}.bmp_output_GPU.png"
    
    cmd = [INPAINT_BIN, image_path, cpu_out, gpu_out]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    
    metrics = {}
    for line in result.stdout.splitlines():
        if line.startswith("METRICS_JSON:"):
            metrics = json.loads(line[len("METRICS_JSON:"):])
            break

    return metrics, cpu_out, gpu_out



@app.route("/")
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


@app.route("/images")
def list_images():
    """List available images in the images directory."""
    exts = {".bmp", ".png", ".jpg", ".jpeg"}
    files = []
    for f in sorted(os.listdir(IMAGES_DIR)):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(f)
    return jsonify(files)


@app.route("/image/<filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/inpaint", methods=["POST"])
def inpaint():
    """
    Accepts:
      - image_path: relative filename in IMAGES_DIR, OR
      - image_data: base64 PNG (uploaded from browser with user mask painted)
      - mask_data:  base64 PNG mask (white = remove)

    Returns JSON with CPU time, GPU time, speedup, quality metrics,
    and base64-encoded result images.
    """
    data = request.get_json()

    if "image_data" in data:
        img_bytes  = base64.b64decode(data["image_data"].split(",")[-1])
        img        = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_name   = f"upload_{int(time.time())}.bmp"
        img_path   = os.path.join(UPLOAD_DIR, img_name)
        img.save(img_path, format="BMP")
    elif "image_path" in data:
        img_path = os.path.join(IMAGES_DIR, data["image_path"])
        if not os.path.exists(img_path):
            return jsonify({"error": "Image not found"}), 404
    else:
        return jsonify({"error": "No image provided"}), 400

    try:
        cpu_time, cpu_out = run_inpaint(img_path, "cpu")
    except Exception as e:
        return jsonify({"error": f"CPU run failed: {e}"}), 500

    gpu_time, gpu_out, gpu_error = None, None, None
    try:
        gpu_time, gpu_out = run_inpaint(img_path, "gpu")
    except Exception as e:
        gpu_error = str(e)

    cpu_img = np.array(Image.open(cpu_out).convert("RGB"))
    metrics = {}
    if gpu_out and os.path.exists(gpu_out):
        gpu_img = np.array(Image.open(gpu_out).convert("RGB"))
        metrics = compute_metrics(cpu_img, gpu_img)

    speedup = round(cpu_time / gpu_time, 3) if gpu_time else None

    response = {
        "cpu_time":   round(cpu_time, 3),
        "gpu_time":   round(gpu_time, 3) if gpu_time else None,
        "speedup":    speedup,
        "gpu_error":  gpu_error,
        "metrics":    metrics,
        "cpu_result": mat_to_b64(cpu_out),
        "gpu_result": mat_to_b64(gpu_out) if gpu_out and os.path.exists(gpu_out) else None,
        "original":   mat_to_b64(img_path),
    }

    return jsonify(response)


if __name__ == "__main__":
    print(f"  Images directory : {os.path.abspath(IMAGES_DIR)}")
    print(f"  Inpaint binary   : {os.path.abspath(INPAINT_BIN)}")
    print(f"  Upload directory : {os.path.abspath(UPLOAD_DIR)}")
    app.run(debug=True, port=5000)