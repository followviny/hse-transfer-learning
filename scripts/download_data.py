"""Download pinned original datasets. Run from the repository root."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile
import urllib.parse
import argparse
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
PV_REV = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
PD_REV = "5467f6012d78d1c446145d5f582da6096f852ae8"
SOURCES = {
    "plantvillage": {
        "url": f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{PV_REV}/data.zip",
        "revision": PV_REV,
        "sha256": "fba30c6a7965e49be94b47a62f8aff6cfb1c35c27f475f22092b56db41745e84",
        "license": "CC BY-SA 3.0 (dataset card)",
    },
    "plantdoc": {
        "url": f"https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/{PD_REV}",
        "revision": PD_REV,
        "license": "CC BY 4.0 (authors repository)",
    },
}


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, target):
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    tmp.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(SOURCES))
    args = parser.parse_args()
    record_path = ROOT / "data/external/sources.json"
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name, spec in SOURCES.items():
        if args.only and args.only != name:
            continue
        if name == "plantdoc":
            tree_path = ROOT / "data/external/plantdoc_tree.json"
            download(f"https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/git/trees/{PD_REV}?recursive=1", tree_path)
            tree = json.loads(tree_path.read_text())
            assert not tree.get("truncated"), "Incomplete source tree"
            members = [x for x in tree["tree"] if x["type"] == "blob"]
            def fetch(member):
                path = member["path"]
                target = ROOT / "data/raw/plantdoc" / path
                url = f"https://raw.githubusercontent.com/pratikkayal/PlantDoc-Dataset/{PD_REV}/" + urllib.parse.quote(path)
                for attempt in range(4):
                    try:
                        download(url, target)
                        payload = target.read_bytes()
                        digest = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
                        if digest != member["sha"]:
                            target.unlink()
                            raise ValueError(f"Git blob mismatch: {path}")
                        return
                    except Exception:
                        if attempt == 3:
                            raise
            with ThreadPoolExecutor(max_workers=12) as pool:
                for i, _ in enumerate(pool.map(fetch, members), 1):
                    if i % 200 == 0:
                        print(f"PlantDoc: {i}/{len(members)} verified", flush=True)
            records[name] = {**spec, "transport": "GitHub raw files, verified against Git blob SHA-1", "extracted_files": len(members)}
            continue
        archive = ROOT / "data/raw" / f"{name}.zip"
        print(f"Download / verify {name}", flush=True)
        download(spec["url"], archive)
        digest = sha256(archive)
        if spec.get("sha256") and digest != spec["sha256"]:
            raise ValueError(f"Checksum mismatch: {archive}")
        dest = ROOT / "data/raw" / name
        count = 0
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                parts = Path(member.filename).parts
                if name == "plantvillage":
                    if "color" not in parts:
                        continue
                    rel = Path(*parts[parts.index("color") + 1:])
                else:
                    rel = Path(*parts[1:])
                if member.is_dir() or str(rel) == ".":
                    continue
                target = dest / rel
                if not target.resolve().is_relative_to(dest.resolve()):
                    raise ValueError("Unsafe archive path")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != member.file_size:
                    with z.open(member) as src, target.open("wb") as out:
                        shutil.copyfileobj(src, out)
                count += 1
        records[name] = {**spec, "archive_sha256": digest, "extracted_files": count}
        print(f"{name}: {count} files", flush=True)
    download(f"https://huggingface.co/datasets/mohanty/PlantVillage/resolve/{PV_REV}/leaf_grouping/leaf-map.json",
             ROOT / "data/external/leaf-map.json")
    existing = json.loads(record_path.read_text()) if record_path.exists() else {}
    record_path.write_text(json.dumps({**existing, **records}, indent=2))


if __name__ == "__main__":
    main()
