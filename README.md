# Agentic File Organizer AI

A production-ready file organization agent using LangGraph, LangChain, FastAPI, and LangSmith for observability.

## Features

- **Agentic Architecture**: Built with LangGraph for stateful agent workflows
- **AI-Powered Classification**: Uses OpenAI GPT for intelligent file categorization
- **Production Ready**: FastAPI backend with PostgreSQL, and Docker
- **Observability**: LangSmith integration for tracing and monitoring
- **Memory System**: Learns from past decisions and can be corrected
- **REST API**: Full-featured API for integration and automation
- **Real-time Updates**: SSE streaming for agent execution
- **Monitoring**: Prometheus metrics and Grafana dashboards

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **AI Framework**: LangChain, LangGraph
- **LLM**: OpenAI GPT-4/GPT-3.5
- **Observability**: LangSmith
- **Database**: PostgreSQL, Redis
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **File Processing**: Watchdog, filetype, python-magic

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose
- OpenAI API key
- LangSmith API key (optional)

### 2. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/file-organizer-agent
cd file-organizer-agent

# Copy environment file
cp .env.example .env

# Edit .env with your API keys
vim .env

# Start services
docker-compose up -d

# Initialize database
docker-compose exec app python -c "from app.database.session import init_db; init_db()"
