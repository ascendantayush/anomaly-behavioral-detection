# Deployment Guide

This guide covers deploying the AI-Powered Behavioral Anomaly Detection system to production. We analyze the architecture first, then provide step-by-step instructions for four deployment options.

---

## Table of Contents

1. [Architecture Analysis](#1-architecture-analysis)
2. [Repository Structure for Deployment](#2-repository-structure-for-deployment)
3. [Option 1: Streamlit Community Cloud](#3-option-1-streamlit-community-cloud)
4. [Option 2: Render](#4-option-2-render)
5. [Option 3: Railway](#5-option-3-railway)
6. [Option 4: Docker Deployment](#6-option-4-docker-deployment)
7. [Environment Variables](#7-environment-variables)
8. [Common Deployment Failures](#8-common-deployment-failures)
9. [Performance Considerations](#9-performance-considerations)
10. [Cost Considerations](#10-cost-cost-considerations)
11. [Security Considerations](#11-security-considerations)
12. [Best Practices](#12-best-practices)
13. [Updating a Deployment](#13-updating-a-deployment)

---

## 1. Architecture Analysis

### Why No Separate Backend Is Required

This project uses **Streamlit** as both the frontend and backend. Streamlit is a full-stack Python framework that:

- Runs Python code directly on the server
- Renders HTML/CSS/JavaScript in the browser automatically
- Handles HTTP requests, WebSocket connections, and session management internally
- Serves Plotly charts, data tables, and interactive widgets natively

There is no separate API server, database, or message queue. The entire pipeline — data generation, model training, inference, and visualization — runs within a single Python process managed by Streamlit's built-in server (Tornado).

### What This Means for Deployment

| Concern | How It's Handled |
|---------|-----------------|
| HTTP server | Streamlit's built-in Tornado server |
| Frontend rendering | Streamlit's React-based client (served automatically) |
| Model training | In-memory, runs on first request, cached by Streamlit |
| Data storage | In-memory only (no database needed) |
| API endpoints | Not required (Streamlit handles client-server communication) |

**Conclusion:** Deploy this as a single Streamlit application. No separate backend, no Docker Compose, no load balancer required for the hackathon or demo use case.

### When You Would Need a Backend

If you later add:
- Real SIEM log ingestion (Kafka, syslog)
- Persistent database storage
- REST API for external integrations
- Multi-user authentication

...then you would add FastAPI or Flask as a separate backend. That is out of scope for this project.

---

## 2. Repository Structure for Deployment

All deployment platforms expect specific files at the repository root. Since the project code lives inside `anomaly_detection/`, you need a root-level entry point.

### Files Required at Repository Root

```
ai-behavioral-anomaly-detection/
├── app.py                    # ← Root-level entry point (NEW)
├── requirements.txt          # ← Root-level dependencies (NEW)
├── anomaly_detection/        # Project package
│   ├── config.py
│   ├── data/
│   ├── features/
│   ├── detection/
│   ├── classification/
│   └── utils/
├── README.md
├── LOCAL_SETUP.md
└── DEPLOYMENT.md
```

### Create the Root-Level Entry Point

Create `app.py` at the repository root (NOT inside `anomaly_detection/`):

```python
"""
Root-level entry point for deployment platforms.
This file simply imports and runs the main application.
"""
import sys
from pathlib import Path

# Ensure the anomaly_detection package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from anomaly_detection.app import main

if __name__ == "__main__":
    main()
```

### Create the Root-Level requirements.txt

Create `requirements.txt` at the repository root:

```
streamlit>=1.32.0
pandas>=2.2.0
numpy>=1.26.0
faker>=24.0.0
scikit-learn>=1.4.0
plotly>=5.18.0
```

This duplicates `anomaly_detection/requirements.txt` at the root level so deployment platforms can find it automatically.

### Fix the Import Path in anomaly_detection/app.py

The `app.py` inside `anomaly_detection/` uses a relative parent path. For deployment, update the path resolution at the top of the file:

```python
# Replace the existing sys.path manipulation with:
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
```

This ensures imports work regardless of the working directory.

---

## 3. Option 1: Streamlit Community Cloud

**Best for:** Quick demo, hackathon presentations, public portfolio projects.
**Cost:** Free tier available.
**Complexity:** Lowest — just push to GitHub.

### Prerequisites

- GitHub account
- Streamlit Community Cloud account (sign up at [share.streamlit.io](https://share.streamlit.io))

### Step-by-Step

1. **Push your code to GitHub:**

```bash
cd /path/to/ai-behavioral-anomaly-detection
git init
git add .
git commit -m "Initial deployment"
git remote add origin https://github.com/your-username/ai-behavioral-anomaly-detection.git
git push -u origin main
```

2. **Connect to Streamlit Community Cloud:**

- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with your GitHub account
- Click **"New app"**
- Select your repository, branch (`main`), and main file (`app.py`)
- Click **"Deploy"**

3. **Wait for deployment:**

Streamlit Community Cloud will:
- Clone your repository
- Install dependencies from `requirements.txt`
- Start the Streamlit server
- Assign a public URL (e.g., `https://your-app-name.streamlit.app`)

4. **Verify:**

Open the assigned URL. The dashboard should load with the full pipeline running.

### Limitations

- Free tier has limited resources (1 GB RAM, shared CPU)
- First load takes 2–5 minutes (cold start)
- No custom domains on free tier
- App sleeps after 15 minutes of inactivity (wakes on next visit)

---

## 4. Option 2: Render

**Best for:** More reliable hosting, custom domains, persistent services.
**Cost:** Free tier available (with limitations), paid plans from $7/month.
**Complexity:** Low.

### Step-by-Step

1. **Push code to GitHub** (same as Option 1).

2. **Create a Render account** at [render.com](https://render.com).

3. **Create a new Web Service:**

- Click **"New +"** → **"Web Service"**
- Connect your GitHub repository
- Configure:
  - **Name:** `ai-anomaly-detection`
  - **Runtime:** Python 3
  - **Build Command:** `pip install -r requirements.txt`
  - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
  - **Port:** 8501

4. **Set environment variables** (Render dashboard → Environment tab):

| Variable | Value | Required |
|----------|-------|----------|
| `PYTHON_VERSION` | `3.12.0` | Yes |
| `STREAMLIT_SERVER_HEADLESS` | `true` | Yes |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` | Recommended |

5. **Deploy:**

Click **"Create Web Service"**. Render will build and deploy your app.

6. **Verify:**

Open the Render-assigned URL (e.g., `https://ai-anomaly-detection.onrender.com`).

### Free Tier Limitations

- Service spins down after 15 minutes of inactivity
- 512 MB RAM, shared CPU
- First request after spin-down takes 30–60 seconds
- 750 hours/month free (enough for demos)

---

## 5. Option 3: Railway

**Best for:** Developer-friendly deployment, good free tier, easy scaling.
**Cost:** Free trial ($5 credit), then $5/month minimum.
**Complexity:** Low.

### Step-by-Step

1. **Push code to GitHub** (same as above).

2. **Create a Railway account** at [railway.app](https://railway.app).

3. **Create a new project:**

- Click **"New Project"** → **"Deploy from GitHub Repo"**
- Select your repository

4. **Configure:**

Railway auto-detects Python projects. If it doesn't, add a `railway.json` at the repository root:

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true",
    "healthcheckPath": "/",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

5. **Set environment variables** (Railway dashboard → Variables tab):

| Variable | Value |
|----------|-------|
| `PYTHON_VERSION` | `3.12` |
| `STREAMLIT_SERVER_HEADLESS` | `true` |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `false` |

6. **Deploy:**

Railway auto-deploys on every push to `main`.

7. **Generate a public domain:**

- Go to **Settings** → **Networking**
- Click **"Generate Domain"**
- Use the assigned `.railway.app` URL

---

## 6. Option 4: Docker Deployment

**Best for:** Self-hosted environments, CI/CD pipelines, reproducible builds.
**Cost:** Free (you provide the infrastructure).
**Complexity:** Medium.

### Dockerfile

Create a `Dockerfile` at the repository root:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
```

### .dockerignore

Create a `.dockerignore` at the repository root:

```
__pycache__
*.pyc
.git
.gitignore
venv/
env/
*.md
screenshots/
```

### Build and Run

```bash
# Build the image
docker build -t anomaly-detection .

# Run the container
docker run -d \
    --name anomaly-detection \
    -p 8501:8501 \
    anomaly-detection

# Verify
docker logs anomaly-detection
```

Open `http://localhost:8501` in your browser.

### Docker Compose (Optional)

For more complex setups, create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  anomaly-detection:
    build: .
    ports:
      - "8501:8501"
    restart: unless-stopped
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose up -d
```

---

## 7. Environment Variables

The following environment variables can be set on any deployment platform:

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Port the server listens on |
| `STREAMLIT_SERVER_ADDRESS` | `localhost` | Bind address (`0.0.0.0` for external access) |
| `STREAMLIT_SERVER_HEADLESS` | `false` | Disable browser auto-open (set `true` for servers) |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | `true` | Disable telemetry (set `false` for privacy) |
| `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` | `200` | Max file upload size in MB |
| `STREAMLIT_THEME_BASE` | `dark` | Dashboard theme (`dark` or `light`) |

**Note:** This project does not use secrets, API keys, or database credentials. All data is generated synthetically in-memory.

---

## 8. Common Deployment Failures

### `No module named 'anomaly_detection'`

**Cause:** The root-level `app.py` doesn't have the correct path setup.

**Fix:** Ensure the root `app.py` adds the project directory to `sys.path`:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent))
```

### `Requirements not found`

**Cause:** The deployment platform can't find `requirements.txt`.

**Fix:** Make sure `requirements.txt` exists at the repository root, not just inside `anomaly_detection/`.

### `Port already in use`

**Cause:** The platform assigns a port via the `$PORT` environment variable.

**Fix:** Use `$PORT` in your start command:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### `Memory exceeded`

**Cause:** The default dataset is too large for the free tier.

**Fix:** Reduce dataset size in `anomaly_detection/config.py`:
```python
num_users: int = 20
days: int = 14
events_per_user_per_day: int = 15
```

### `Build timeout`

**Cause:** pip install is slow (scikit-learn and streamlit are large packages).

**Fix:**
- Pin exact versions to avoid downloading multiple candidate versions
- Use `--no-cache-dir` in pip install
- Consider using a `requirements.txt` with pinned versions:
  ```
  streamlit==1.32.0
  pandas==2.2.0
  numpy==1.26.4
  faker==24.2.0
  scikit-learn==1.4.0
  plotly==5.18.0
  ```

### `Application startup timeout`

**Cause:** Streamlit takes too long to start (model training on first request).

**Fix:** The Streamlit cache handles this after the first request. For platforms with strict startup timeouts, the initial data generation will complete within the timeout on subsequent visits.

---

## 9. Performance Considerations

### Memory Usage

| Component | Approximate RAM |
|-----------|----------------|
| Streamlit server | ~150 MB |
| Synthetic data (50 users, 30 days) | ~50 MB |
| Feature matrix (43 columns) | ~30 MB |
| Isolation Forest model | ~20 MB |
| Markov model | ~10 MB |
| **Total** | **~260 MB** |

For free tiers with 512 MB RAM, this leaves adequate headroom.

### CPU Usage

- **Data generation:** CPU-intensive for 1–3 seconds
- **Feature engineering:** CPU-intensive for 2–5 seconds
- **Model training:** CPU-intensive for 3–10 seconds
- **Dashboard rendering:** Minimal (cached after first load)

Total initial CPU burst: ~15 seconds. After caching, dashboard is served with negligible CPU.

### Cold Start Optimization

To reduce cold-start time on platforms that sleep idle services:

1. **Reduce dataset size** (see Memory section above)
2. **Pre-generate data** and save to disk (adds persistence requirement)
3. **Use a paid tier** that doesn't sleep

### Caching Strategy

The dashboard uses `@st.cache_data` for both data generation and model training. This means:
- First visit: full pipeline runs (~15 seconds)
- Subsequent visits: instant (served from cache)
- Cache persists across browser sessions
- Cache is invalidated only when code changes

---

## 10. Cost Considerations

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| Streamlit Community Cloud | 1 GB RAM, limited hours | N/A (free only) | Demos, portfolios |
| Render | 512 MB, 750 hrs/month | $7/month (512 MB) | Reliable hosting |
| Railway | $5 trial credit | $5/month minimum | Development |
| Docker (self-hosted) | Free (you provide infra) | Varies | Full control |

### Cost Optimization

- **Streamlit Community Cloud** is free and sufficient for hackathon demos
- **Render free tier** works well for longer-term demos (app sleeps after 15 min idle)
- **Railway** is cost-effective for always-on services
- **Docker on a $5/month VPS** (DigitalOcean, Hetzner) gives full control

---

## 11. Security Considerations

### Current Security Posture

This project generates synthetic data entirely in-memory. There are:

- **No real credentials** or secrets
- **No database** connections
- **No external API calls**
- **No user authentication** required

### Security Recommendations for Production

If deploying with real data:

1. **Add authentication** — Use Streamlit's built-in auth or a reverse proxy (nginx) with basic auth
2. **Use HTTPS** — All platforms above provide HTTPS by default
3. **Restrict network access** — Deploy behind a VPN or firewall for internal use
4. **Don't log sensitive data** — Ensure no real credentials appear in logs
5. **Use environment variables** — Never hardcode secrets in source code
6. **Rate limiting** — Add rate limiting if exposed to the internet

---

## 12. Best Practices

1. **Keep `requirements.txt` at the root** — Deployment platforms look for it there
2. **Use a root-level `app.py` entry point** — Platform detection is more reliable
3. **Pin dependency versions** — Prevents breaking changes from upstream updates
4. **Set `server.headless=true`** — Required for headless server environments
5. **Disable usage stats** — `browser.gatherUsageStats=false` for privacy
6. **Use `@st.cache_data`** — Already implemented; prevents redundant computation
7. **Test locally first** — Always verify with `streamlit run app.py` before deploying
8. **Monitor memory** — Keep dataset size appropriate for the deployment tier
9. **Use Docker for reproducibility** — Eliminates "works on my machine" issues
10. **Tag releases** — Use git tags for deployed versions (`git tag v1.0.0`)

---

## 13. Updating a Deployment

### Streamlit Community Cloud

Push to `main` — deployment auto-triggers:

```bash
git add .
git commit -m "Update: description of changes"
git push origin main
```

### Render

Push to `main` — deployment auto-triggers. For manual redeploy:

- Go to Render dashboard → your service → **"Manual Deploy"** → **"Deploy latest commit"**

### Railway

Push to `main` — deployment auto-triggers.

### Docker

Rebuild and redeploy:

```bash
docker build -t anomaly-detection .
docker stop anomaly-detection
docker rm anomaly-detection
docker run -d --name anomaly-detection -p 8501:8501 anomaly-detection
```

### Version Tagging

Tag each deployed version:

```bash
git tag -a v1.0.0 -m "Initial deployment"
git push origin v1.0.0
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Deploy to Streamlit Cloud | Push to GitHub → connect at share.streamlit.io |
| Deploy to Render | Push to GitHub → New Web Service → set start command |
| Deploy to Railway | Push to GitHub → New Project → Deploy from Repo |
| Deploy with Docker | `docker build -t app . && docker run -d -p 8501:8501 app` |
| Update deployment | `git push origin main` (auto-triggers on all platforms) |
| Check deployment logs | Platform dashboard → Logs tab |
| Scale up | Platform dashboard → Settings → Resources → increase RAM/CPU |
