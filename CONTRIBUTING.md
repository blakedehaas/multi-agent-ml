# Contributing & Collaboration Guide

How this project is developed, how the tools fit together, and what each collaborator needs to set up.

---

## Stack

| Tool | Role |
|------|------|
| **GitHub** | Source of truth for all `.py` modules and the notebook |
| **Google Colab** | Compute — GPU training sessions |
| **Google Drive** | Persistent outputs — checkpoints, figures |
| **Weights & Biases** | Experiment tracking and live metric streaming |

---

## Collaboration Loop

All source code lives on GitHub. Drive stores outputs only. The workflow is:

```
  Edit .py files locally
        ↓
  git push → GitHub (implementation branch)
        ↓
  Colab: !git pull origin implementation
        ↓
  autoreload picks up changes — no restart needed
```

For notebook-level changes (adding/restructuring cells), edit the `.ipynb` file and push.

**Rule of thumb**: batch notebook edits. Don't iterate one cell at a time — accumulate structural changes and pull them all at once.

---

## First-Time Colab Setup (per collaborator)

### 1. Google Drive
Checkpoints and figures are saved to a shared Drive folder so both collaborators see the same outputs. The notebook path cell has `DRIVE_ROOT` set to the shared folder — update the commented line to match your own Drive layout if needed.

**One-time setup for the non-owner collaborator:**
1. Owner shares the `Final_Project` folder — right-click → Share → add collaborator's email with Editor access
2. Collaborator opens **Shared with me** in Google Drive
3. Right-click `Final_Project` → **Organize** → **Add shortcut**
4. Place the shortcut at **My Drive** (root level, not inside any subfolder)

The shortcut must be at the root of My Drive so that `/content/drive/MyDrive/Final_Project` resolves to the same path for both collaborators. Checkpoints and figures saved by either person land in the same folder.

### 2. Weights & Biases
Each collaborator logs in with their own W&B account:
```python
import wandb
wandb.login()  # paste your API key from wandb.ai/settings
```

Both accounts should be added to the shared W&B project (`swarm-optimization`) so runs from either collaborator appear in the same dashboard.

### 3. GitHub access
The repo is at `github.com/blakedehaas/multi-agent-ml`. Both collaborators need read and push access for `git clone`.

---

## Branch Structure

| Branch | Purpose |
|--------|---------|
| `main` | Stable, clean — merged to at milestones |
| `implementation` | Active development — all day-to-day work happens here |


---

## Output Conventions

| Output | Location | Persisted? |
|--------|----------|------------|
| Model checkpoints (`.pt`) | `DRIVE_ROOT/experiments/checkpoints/` | ✅ Drive |
| Figures (`.png`) | `DRIVE_ROOT/experiments/checkpoints/` | ✅ Drive |
| CIFAR-10 data | `/content/data/` (Colab local) | ❌ Re-downloads each session |
| W&B logs | wandb.ai cloud | ✅ W&B |
| Source code | GitHub | ✅ Git |

---

## Saving Work Back to GitHub from Colab

If you edit a `.py` file directly in Colab's file browser:

```bash
git add path/to/file.py
git commit -m "your message"
git push origin implementation
```

If you add or edit notebook cells directly in Colab:

`File → Save a copy in GitHub` → select `implementation` branch → write a commit message
