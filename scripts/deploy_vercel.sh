#!/bin/bash
# Deploy to Vercel and test the deployment

set -e  # Exit on error

echo "🚀 Deploying to Vercel..."
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if logged in
if ! vercel whoami &> /dev/null; then
    echo "❌ Not logged in to Vercel. Please run: vercel login"
    exit 1
fi

echo "✅ Vercel CLI ready"
echo ""

# Run local tests first
echo "📋 Running local tests..."
python3 scripts/test_vercel_node_logic.py
if [ $? -ne 0 ]; then
    echo "❌ Local tests failed. Fix issues before deploying."
    exit 1
fi

echo ""
echo "✅ Local tests passed"
echo ""

# Deploy to production
echo "🌐 Deploying to Vercel production..."
vercel --prod

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deployment successful!"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Test the /api/health endpoint"
    echo "   2. Test the /api/chat endpoint with the three conversation turns"
    echo "   3. Verify environment variables are set in Vercel dashboard"
    echo ""
    echo "🧪 Test commands:"
    echo "   curl https://your-app.vercel.app/api/health"
    echo "   curl -X POST https://your-app.vercel.app/api/chat \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"query\": \"\", \"session_id\": \"test-001\"}'"
else
    echo "❌ Deployment failed"
    exit 1
fi
