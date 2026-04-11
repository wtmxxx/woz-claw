# Skill Zip Upload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在配置中心的 Skills 区增加 zip 上传入口，上传后自动解压、按 `SKILL.md` 的 `name` 命名、并自动启用到当前用户配置。

**Architecture:** 后端新增独立上传接口和一组小的文件处理辅助函数，负责 zip 校验、临时解压、`SKILL.md` frontmatter 读取、目录覆盖和 `skills.yaml` 更新。前端只加文件选择与上传按钮，成功后重新加载 settings，让列表立即反映新 skill。

**Tech Stack:** Python 3.11, FastAPI, PyYAML, pytest, vanilla HTML/JS

---

### Task 1: 写上传接口回归测试（先红）

**Files:**
- Modify: `tests/test_app_api.py`

- [ ] **Step 1: 写失败测试，验证 zip 上传后会解压并自动启用**

```python
from io import BytesIO
from zipfile import ZipFile


def test_upload_skill_zip_extracts_and_enables_skill(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "multi-search-engine/SKILL.md",
            "---\nname: multi-search-engine\ndescription: test\n---\n# demo\n",
        )
        archive.writestr("multi-search-engine/README.md", "demo")

    response = client.post(
        "/api/skills/upload",
        data={"user_id": "u1"},
        files={"file": ("multi-search-engine-2.0.1.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 200
    assert (skills_root / "u1" / "multi-search-engine" / "SKILL.md").exists()


def test_upload_skill_zip_falls_back_to_zip_name_without_version(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "multi-search-engine-2.0.1/SKILL.md",
            "---\ndescription: test\n---\n# demo\n",
        )

    response = client.post(
        "/api/skills/upload",
        data={"user_id": "u1"},
        files={"file": ("multi-search-engine-2.0.1.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 200
    assert (skills_root / "u1" / "multi-search-engine" / "SKILL.md").exists()


def test_upload_skill_zip_overwrites_existing_skill(monkeypatch, tmp_path):
    skills_root = tmp_path / "skills"
    target_dir = skills_root / "u1" / "multi-search-engine"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "SKILL.md").write_text("---\nname: old\n---\nold", encoding="utf-8")
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "multi-search-engine/SKILL.md",
            "---\nname: multi-search-engine\ndescription: test\n---\n# demo\n",
        )

    response = client.post(
        "/api/skills/upload",
        data={"user_id": "u1"},
        files={"file": ("multi-search-engine.zip", buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 200
    assert (target_dir / "SKILL.md").read_text(encoding="utf-8").startswith("---\nname: multi-search-engine")
```

- [ ] **Step 2: 运行单测确认失败（RED）**

Run: `pytest -q tests/test_app_api.py::test_upload_skill_zip_extracts_and_enables_skill tests/test_app_api.py::test_upload_skill_zip_falls_back_to_zip_name_without_version`
Expected: FAIL，当前还没有上传接口。

- [ ] **Step 3: 提交测试变更**

```bash
git add tests/test_app_api.py
git commit -m "test: add skill zip upload regression coverage"
```

### Task 2: 实现后端上传与配置落盘

**Files:**
- Modify: `src/wozclaw/app.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_app_api.py`

- [ ] **Step 1: 实现 zip 上传端点与辅助函数**

```python
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pathlib import Path
from zipfile import ZipFile
import re
import shutil
import tempfile


def _normalize_skill_zip_name(zip_filename: str) -> str:
    stem = Path(zip_filename).stem.strip()
    normalized = re.sub(r"(?i)[-_]?v?\d+(?:\.\d+)+$", "", stem).strip("-_. ")
    return normalized or stem


def _read_skill_name(skill_md_path: Path) -> str | None:
    text = skill_md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    frontmatter = yaml.safe_load(parts[1])
    if not isinstance(frontmatter, dict):
        return None
    name = str(frontmatter.get("name", "")).strip()
    return name or None


def _sync_user_skill_toggle(user_id: str, skill_name: str) -> None:
    path = _user_skills_yaml_path(user_id)
    existing = _load_user_skill_toggles(user_id)
    merged: list[dict[str, Any]] = []
    seen = False
    for item in existing:
        if item["name"] == skill_name:
            merged.append({"name": skill_name, "enabled": True})
            seen = True
        else:
            merged.append(item)
    if not seen:
        merged.append({"name": skill_name, "enabled": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"skills": merged}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@app.post("/api/skills/upload")
async def upload_skill_zip(user_id: str = Form(...), file: UploadFile = File(...)) -> dict[str, Any]:
    user_text = user_id.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="user_id is required")
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="only zip files are supported")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / Path(file.filename).name
        archive_path.write_bytes(await file.read())

        with ZipFile(archive_path) as archive:
            archive.extractall(temp_root)

        skill_md_candidates = list(temp_root.rglob("SKILL.md"))
        if not skill_md_candidates:
            raise HTTPException(status_code=400, detail="SKILL.md not found in archive")

        skill_md_path = skill_md_candidates[0]
        skill_dir = skill_md_path.parent
        skill_name = _read_skill_name(skill_md_path) or _normalize_skill_zip_name(file.filename)
        if not skill_name:
            raise HTTPException(status_code=400, detail="skill name is required")

        target_dir = _skills_root_dir() / user_text / skill_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(skill_dir), str(target_dir))

    _sync_user_skill_toggle(user_text, skill_name)
    return {"ok": True, "name": skill_name, "dir": str(target_dir)}
```

- [ ] **Step 2: 补齐依赖声明**

```toml
dependencies = [
	"fastapi>=0.115.0",
	"jinja2>=3.1.4",
	"agentscope>=1.0.18",
	"pydantic>=2.9.0",
	"pyyaml>=6.0.0",
	"python-multipart>=0.0.20",
	"pywin32>=311; platform_system == 'Windows'",
	"python-dotenv>=1.0.1",
	"uvicorn[standard]>=0.30.6",
]
```

- [ ] **Step 3: 运行后端测试确认通过**

Run: `pytest -q tests/test_app_api.py`
Expected: PASS，上传测试、现有 settings 测试都通过。

- [ ] **Step 4: 提交后端变更**

```bash
git add src/wozclaw/app.py pyproject.toml tests/test_app_api.py
git commit -m "feat: add skill zip upload api"
```

### Task 3: 增加配置中心上传入口

**Files:**
- Modify: `src/wozclaw/static/index.html`
- Modify: `tests/test_frontend_rendering.py`

- [ ] **Step 1: 写前端快照测试，验证 Skills 区存在上传控件**

```python
def test_frontend_has_skill_upload_controls() -> None:
    html_path = Path("src/wozclaw/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert 'id="skillUploadFile"' in html
    assert 'id="skillUploadBtn"' in html
    assert "上传 Skill ZIP" in html
```

- [ ] **Step 2: 在 Skills 面板加入文件选择和上传按钮**

```html
<div class="settings-group">
    <label for="skillUploadFile">上传 Skill ZIP</label>
    <input id="skillUploadFile" type="file" accept=".zip,application/zip" />
    <button id="skillUploadBtn" class="secondary-btn" type="button">上传 Skill ZIP</button>
</div>
```

- [ ] **Step 3: 接上上传后的刷新逻辑**

```javascript
async function uploadSkillZip() {
    const file = skillUploadFileEl.files && skillUploadFileEl.files[0];
    if (!file) throw new Error('请选择 zip 文件');

    const formData = new FormData();
    formData.append('user_id', userIdEl.value.trim());
    formData.append('file', file);

    const res = await fetch('/api/skills/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '上传失败');

    await loadSettings();
}
```

- [ ] **Step 4: 运行前端测试确认通过**

Run: `pytest -q tests/test_frontend_rendering.py`
Expected: PASS，上传控件和既有布局断言全部通过。

- [ ] **Step 5: 提交前端变更**

```bash
git add src/wozclaw/static/index.html tests/test_frontend_rendering.py
git commit -m "feat: add skill zip upload controls"
```

### Task 4: 全量验证与收尾

**Files:**
- Test: `tests/test_app_api.py`
- Test: `tests/test_frontend_rendering.py`
- Test: `tests/test_agent_prompt.py`

- [ ] **Step 1: 运行覆盖相关改动的测试**

Run: `pytest -q tests/test_app_api.py tests/test_frontend_rendering.py tests/test_agent_prompt.py`
Expected: 全部 PASS。

- [ ] **Step 2: 检查工作区变更并确认没有遗漏文件**

Run: `git status --short`
Expected: 只包含本次 feature 相关变更。

- [ ] **Step 3: 最终提交**

```bash
git add src pyproject.toml tests docs/superpowers
git commit -m "feat: support skill zip upload in settings"
```