"""Build an allowlisted GitHub sharing bundle using only the Python standard library."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "safety-infer-demo"
DIST = ROOT / "dist"
UPLOAD_ROOT = DIST / "github-upload"
STAGING = UPLOAD_ROOT / PACKAGE_NAME
ARCHIVE = DIST / f"{PACKAGE_NAME}-github.zip"
MANIFEST = "PACKAGE_MANIFEST.json"

ROOT_FILES = """
.env.example .gitignore README.md INSTALLATION_GUIDE.md OPERATION_MODEL.md
GITHUB_UPLOAD.md DATA_MODEL_NOTES.md THIRD_PARTY_NOTICES.md
requirements.txt requirements-operations.txt requirements-prediction.txt requirements-dev.txt
app.py inference_service.py risk_scoring.py runtime_config.py
operation_models.py operation_service.py operation_knowledge.json digital_twin_service.py
prediction_api.py prediction_service.py streamlit_app.py workbench_ui.py
CNN-LSTM-AT-predict.py C_L_A.h5 inputdata.csv start_operations.ps1
test_digital_twin.py test_llm_decision.py test_neo4j.py test_operations.py
test_operation_api.py test_operation_inputs.mjs test_streamlit_workbench.py test_tower_graph.py
""".split()
OTHER_FILES = [
    ".streamlit/config.toml", "assets/workbench.css",
    "examples/README.md", "examples/obstacle-input.json", "examples/validation-template.csv",
    "scripts/build_github_package.py",
]
GRAPH_FILES = [f"Knowledge Graph/{name}.csv" for name in (
    "SafetyFactor", "Indicator", "SafetyRisk", "SafetyAccident", "ConstructionStage",
    "HAS_INDICATOR", "ACTIVE_IN", "CAUSES", "LEADS_TO",
)]
TWIN_FILES = [f"digital_twin/{name}" for name in (
    "index.html", "legacy.html", "operations.html", "main.mjs", "operations.mjs",
    "operation_scene.mjs", "styles.css", "operations.css",
    "vendor/package/LICENSE", "vendor/package/build/three.module.mjs",
    "vendor/package/build/three.core.mjs", "vendor/package/examples/jsm/controls/OrbitControls.mjs",
)]

# Check patterns without ever printing matched content.
SECRET_PATTERNS = [
    ("provider token", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("AWS access key", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Google API key", re.compile(rb"\bAIza[A-Za-z0-9_-]{30,}\b")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("credential URL", re.compile(rb"(?:bolt|neo4j|https?)://[^\s/:]+:[^\s/@]+@")),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(base: Path, relative: str) -> Path:
    target = base / relative
    if target.is_symlink() or not target.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Unsafe package path: {relative}")
    return target


def check_source(relative: str, data: bytes) -> None:
    for label, pattern in SECRET_PATTERNS:
        match = pattern.search(data)
        if match:
            line = data[:match.start()].count(b"\n") + 1
            raise ValueError(f"Review required: {relative}:{line} ({label}); value withheld")
    if relative.endswith('.py'):
        ast.parse(data.decode('utf-8-sig'), filename=relative)
    if relative == '.env.example':
        for row in data.decode('utf-8-sig').splitlines():
            if row.strip().startswith('DEEPSEEK_API_KEY=') and row.split('=', 1)[1].strip():
                raise ValueError('.env.example must contain an empty API key')


def main() -> None:
    files = sorted(set(ROOT_FILES + OTHER_FILES + GRAPH_FILES + TWIN_FILES))
    payloads = {}
    for relative in files:
        source = safe_path(ROOT, relative)
        data = source.read_bytes()
        check_source(relative, data)
        payloads[relative] = data
    manifest = {
        "package": PACKAGE_NAME, "manifest_format": 1,
        "manifest_self_excluded": True,
        "files": [{"path": name, "bytes": len(data), "sha256": sha256(data)}
                  for name, data in payloads.items()],
    }
    payloads[MANIFEST] = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')

    DIST.mkdir(exist_ok=True)
    # Only replace this script's dedicated output directory, after containment checks.
    resolved_output = UPLOAD_ROOT.resolve()
    if (resolved_output.parent != DIST.resolve() or DIST.resolve().parent != ROOT.resolve()
            or not resolved_output.is_relative_to(ROOT.resolve()) or UPLOAD_ROOT.is_symlink()
            or resolved_output.name != 'github-upload'):
        raise ValueError('Unsafe release directory')
    if UPLOAD_ROOT.exists():
        if not (STAGING / MANIFEST).is_file():
            raise ValueError('Refusing to replace an unrecognized upload directory')
        shutil.rmtree(UPLOAD_ROOT)
    STAGING.mkdir(parents=True)
    for name, data in payloads.items():
        target = safe_path(STAGING, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    temporary_archive = DIST / (ARCHIVE.name + '.tmp')
    with zipfile.ZipFile(temporary_archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted(payloads.items()):
            entry = zipfile.ZipInfo(f'{PACKAGE_NAME}/{name}', date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, data)
    with zipfile.ZipFile(temporary_archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for name, data in payloads.items():
            if bundle.read(f'{PACKAGE_NAME}/{name}') != data:
                raise ValueError(f'Archive content differs: {name}')
    temporary_archive.replace(ARCHIVE)
    checksum = sha256(ARCHIVE.read_bytes())
    ARCHIVE.with_suffix('.zip.sha256').write_text(f'{checksum}  {ARCHIVE.name}\n', encoding='utf-8')
    print(json.dumps({
        "archive": str(ARCHIVE.relative_to(ROOT)), "upload_folder": str(STAGING.relative_to(ROOT)),
        "files_including_manifest": len(payloads), "archive_bytes": ARCHIVE.stat().st_size,
        "sha256": checksum, "checks": "allowlist, secret patterns, Python syntax, manifest, ZIP CRC and content",
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
