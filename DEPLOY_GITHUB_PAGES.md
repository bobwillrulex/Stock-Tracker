# Deploying Stock Tracker to GitHub Pages

This project is already set up to publish `docs/` with GitHub Actions (`.github/workflows/deploy-pages.yml`).

## 1) One-time GitHub setup

1. Push this branch to GitHub.
2. In your GitHub repo, go to **Settings → Pages**.
3. Under **Build and deployment**, set:
   - **Source**: `GitHub Actions`
4. In **Settings → Actions → General**, ensure Actions are enabled.
5. In **Settings → Actions → Workflow permissions**, allow:
   - `Read and write permissions` (recommended for repositories where workflows need to publish pages).

After this, any push that updates `docs/**` triggers the deploy workflow and updates the site.

## 2) First publish

A starter page already exists at `docs/index.html`.

- Push your code once:
  ```bash
  git push origin <your-branch>
  ```
- Open **Actions** tab and wait for **Deploy GitHub Pages** to succeed.
- Your site URL will appear in the workflow summary and in **Settings → Pages**.

## 3) Automatic updates from your 24/7 PC

Your `run_daily()` flow in `main.py` does this automatically:

1. Writes latest recommendations to `buy_signals.csv`.
2. Regenerates `docs/index.html`.
3. Commits and pushes report changes (if `STOCK_TRACKER_AUTO_PUSH` is enabled).

### Environment variable

- `STOCK_TRACKER_AUTO_PUSH=1` (default) → auto commit + push enabled.
- `STOCK_TRACKER_AUTO_PUSH=0` → disables auto push.

## 4) Run daily manually or scheduled

### Manual run

```bash
python main.py
```

### Cron example (Linux)

Run every weekday at 4:10 PM:

```cron
10 16 * * 1-5 cd /path/to/Stock-Tracker && /usr/bin/python3 main.py >> scan.log 2>&1
```

## 5) Authentication for auto-push from your PC

Because the script runs `git push`, your machine must already be authenticated.

Recommended options:

- **SSH key** (best for a 24/7 machine):
  - Add SSH key to GitHub account.
  - Use `git@github.com:<owner>/<repo>.git` remote.
- **Fine-grained PAT** (HTTPS remote):
  - Token needs repo content write access.
  - Store credentials with a credential manager.

## 6) Verify it is working

After a scan run:

```bash
git log -1 --oneline
git status --short
```

Expected:

- Latest commit message like `chore: update stock recommendations (...)`
- Clean working tree
- A successful **Deploy GitHub Pages** workflow run on GitHub

## Troubleshooting

- **Pages workflow not starting**: make sure push changed files under `docs/`.
- **Workflow fails with permissions**: check repo Actions/Page permissions.
- **`git push` fails on PC**: fix SSH/PAT authentication on that machine.
- **No recommendations displayed**: scanner produced no buy signals; page shows empty-state row.
