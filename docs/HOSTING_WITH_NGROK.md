# Hosting the FYP App with ngrok Tunnel

This guide explains how to host your FYP (Final Year Project) app with an ngrok tunnel so you can share it with others without deploying to the cloud.

## Why ngrok?

- **No cloud deployment needed** - Runs locally on your machine
- **Instant sharing** - Get a public URL in seconds
- **Works with existing setup** - Uses your local databases
- **Free tier available** - ngrok offers a free tier

## Prerequisites

1. **ngrok account** - Sign up at https://dashboard.ngrok.com
2. **Docker & Docker Compose** installed
3. **ngrok auth token** from your dashboard

---

## Quick Start (Recommended)

### Option 1: Using the Startup Script

```bash
# 1. Navigate to your project directory
cd /Users/brianho/FYP

# 2. Run the startup script with your ngrok token
./start_with_ngrok.sh YOUR_NGROK_AUTH_TOKEN

# Example:
./start_with_ngrok.sh your_token_here
```

The script will:
- Start PostgreSQL and Neo4j
- Start ngrok tunnel on port 8501
- Start Streamlit app
- Display your public URL

### Option 2: Using Docker Compose

```bash
# 1. Set your ngrok token
export NGROK_AUTHTOKEN=your_token_here

# 2. Start all services including Streamlit with ngrok
docker-compose up -d streamlit

# 3. Check logs for the public URL
docker logs fyp-streamlit

# Or check ngrok dashboard
# Open http://localhost:4040 in your browser
```

---

## Manual Setup (Step by Step)

### Step 1: Get Your ngrok Token

1. Go to https://dashboard.ngrok.com/signup
2. Create a free account
3. Go to "Your Authtoken" section
4. Copy your auth token

### Step 2: Configure ngrok

```bash
# Install ngrok (if not installed)
brew install ngrok  # macOS
# or
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.zip -o ngrok.zip && unzip ngrok.zip

# Configure your auth token
ngrok config add-authtoken YOUR_TOKEN
```

### Step 3: Start the Databases

```bash
# Start PostgreSQL and Neo4j
docker-compose up -d postgres neo4j

# Wait for them to be ready
docker-compose ps
```

### Step 4: Start ngrok and Streamlit

```bash
# Terminal 1: Start ngrok tunnel
ngrok http 8501

# Terminal 2: Start Streamlit
cd POC/streamlit
streamlit run app.py
```

Or run both in one command:

```bash
# Start ngrok in background, then streamlit
ngrok http 8501 &
streamlit run POC/streamlit/app.py
```

---

## Accessing the App

After starting, you'll have access at:

| Service | URL |
|--------|-----|
| **Public (ngrok)** | Check ngrok output or http://localhost:4040 |
| **Local** | http://localhost:8501 |
| **ngrok Dashboard** | http://localhost:4040 |

### Sharing with Others

1. Open http://localhost:4040
2. Click the public URL shown
3. Share that URL with anyone!

---

## Troubleshooting

### "Connection Refused" Errors

Make sure all services are running:
```bash
docker-compose ps
```

### ngrok Shows "Failed to start tunnel"

- Check your ngrok token is correct
- Make sure port 8501 is not in use: `lsof -i :8501`

### Database Connection Issues

The Streamlit app needs to connect to PostgreSQL and Neo4j. Make sure:
```bash
# Check database status
docker-compose ps

# Check logs
docker-compose logs postgres
docker-compose logs neo4j
```

### ngrok Token Not Working

1. Get a new token from https://dashboard.ngrok.com
2. Reconfigure: `ngrok config add-authtoken YOUR_NEW_TOKEN`
3. Restart ngrok: `pkill ngrok && ngrok http 8501`

---

## Environment Variables

If using Docker Compose, you can set these in `.env`:

```bash
# .env file
NGROK_AUTHTOKEN=your_ngrok_token_here
DEEPSEEK_API_KEY=your_deepseek_key
POSTGRES_PASSWORD=airflow
NEO4J_PASSWORD=SecureNeo4jPass2025!
```

---

## Security Notes

⚠️ **Important when sharing:**

1. **Your DeepSeek API key is exposed** - Anyone with the URL can use your API key
2. **Your local network is accessible** - ngrok creates a tunnel to your machine
3. **No authentication on the app** - Consider adding password protection

### Recommendations:

- Use a separate DeepSeek API key for sharing (set limits in DeepSeek dashboard)
- Don't share sensitive information
- Delete the tunnel when done sharing
- Monitor your API usage

---

## Alternative: Streamlit Cloud (No ngrok)

If you want a permanent URL without ngrok:

1. Push your code to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set environment variables in Streamlit Cloud settings

**Limitation:** You'll need cloud-hosted PostgreSQL and Neo4j (not local).

---

## File Changes Summary

| File | Change |
|------|--------|
| `docker/requirements.txt` | Added `pyngrok>=7.2.0` |
| `docker-compose.yml` | Added `streamlit` service with ngrok |
| `start_with_ngrok.sh` | New startup script for easy deployment |

---

## Quick Reference

```bash
# Start everything with ngrok
cd /Users/brianho/FYP
./start_with_ngrok.sh YOUR_NGROK_TOKEN

# Stop everything
docker-compose down
pkill ngrok

# Check ngrok status
curl localhost:4040/api/tunnels

# View streamlit logs
docker logs fyp-streamlit
```

---

**Need Help?**
- ngrok docs: https://ngrok.com/docs
- Streamlit docs: https://docs.streamlit.io
- Check logs at http://localhost:4040
