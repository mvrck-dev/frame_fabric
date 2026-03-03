#!/bin/bash

echo "🚀 Starting VisionPhase Application..."

# Cleanup old processes and locks just in case
lsof -ti:3000 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
rm -f frontend/.next/dev/lock 2>/dev/null

echo "📦 Booting FastAPI Backend (SAM ML Engine) on port 8000..."
cd backend
conda run --no-capture-output -n vp uvicorn main:app --port 8000 &
BACKEND_PID=$!
cd ..

echo "🌐 Booting Next.js Frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "========================================================"
echo "✅ Both servers are starting up!"
echo "➡️ Frontend UI: http://localhost:3000"
echo "➡️ Backend API: http://localhost:8000"
echo "⚠️ Note: The backend may take ~20-30s to load the SAM weights"
echo "🛑 Press Ctrl+C in this terminal to stop both servers."
echo "========================================================"

# Trap allows us to kill both background processes if we Ctrl+C this script
trap 'echo "\nStopping servers..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' SIGINT SIGTERM EXIT

# Wait for both background processes
wait $BACKEND_PID $FRONTEND_PID

