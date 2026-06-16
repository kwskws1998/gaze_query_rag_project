# Git Push Checklist

This repository should only push source code, configs, scripts, tests, and lightweight docs.

Large local files are ignored by `.gitignore`:

```text
OneStop-Eye-Movements/
artifacts/
data/
*.pdf
*.zip
*.csv.zip
*.safetensors
*.bin
*.pt
*.pth
```

## Initialize And Push

```bash
git init
git add .gitignore GIT_PUSH.md SETUP_GPU.md pyproject.toml setup.py requirements.txt requirements-cu121.txt configs scripts src tests project_overview.md implementation_plan.md
git status --short
git commit -m "Initial gaze-query RAG prototype"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Check For Large Tracked Files Before Push

```bash
git ls-files -s
```

If you accidentally staged a large ignored file:

```bash
git rm --cached <PATH>
git status --short
```

Do not use `git add .` until after checking `.gitignore` behavior on the server.
