#!/bin/bash

echo "🧪 Testing Live Vercel API Endpoints"
echo "===================================="
echo ""

API_BASE="https://noahsaiassistant.vercel.app"

echo "1. Testing /api/chat endpoint..."
curl -X POST "$API_BASE/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is your Python experience?",
    "role": "Hiring Manager (technical)",
    "session_id": "test-live-123"
  }' | jq '.' || echo "Response received (no jq installed)"

echo ""
echo ""
echo "2. Testing /api/feedback endpoint..."
curl -X POST "$API_BASE/api/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "test_msg_123",
    "rating": 5,
    "comment": "Testing live API!",
    "contact_requested": false
  }' | jq '.' || echo "Response received (no jq installed)"

echo ""
echo ""
echo "✅ Live API test complete!"
echo ""
echo "Your API endpoints:"
echo "  - POST $API_BASE/api/chat"
echo "  - POST $API_BASE/api/email"
echo "  - POST $API_BASE/api/feedback"
echo "  - POST $API_BASE/api/confess"
