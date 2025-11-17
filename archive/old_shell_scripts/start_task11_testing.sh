#!/bin/bash
# Quick start script for Task 11: Local Streamlit Testing

set -e  # Exit on error

echo "======================================================================"
echo "Task 11: Intelligent Resume Distribution - Testing & Deployment"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check environment variables
echo -e "${YELLOW}Step 1: Checking environment variables...${NC}"
REQUIRED_VARS=("OPENAI_API_KEY" "SUPABASE_URL" "SUPABASE_SERVICE_ROLE_KEY" "RESEND_API_KEY" "TWILIO_ACCOUNT_SID" "TWILIO_AUTH_TOKEN" "TWILIO_PHONE_NUMBER" "NOAH_PHONE_NUMBER")

MISSING_VARS=()
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    else
        echo -e "  ${GREEN}✓${NC} $var is set"
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}✗ Missing environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "    - $var"
    done
    echo ""
    echo "Please set these in your .env file or export them:"
    echo "  export OPENAI_API_KEY='sk-...'"
    echo "  export SUPABASE_URL='https://...'"
    echo "  etc."
    exit 1
fi

echo -e "${GREEN}✓ All required environment variables are set${NC}"
echo ""

# Step 2: Run automated tests
echo -e "${YELLOW}Step 2: Running automated tests...${NC}"
python3 -m pytest tests/test_conversation_quality.py tests/test_documentation_alignment.py tests/test_resume_distribution.py -v

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All automated tests passed${NC}"
else
    echo -e "${RED}✗ Some tests failed. Please fix before proceeding.${NC}"
    exit 1
fi
echo ""

# Step 3: Open testing guides
echo -e "${YELLOW}Step 3: Opening testing guides...${NC}"
echo "  - STREAMLIT_TESTING_GUIDE.md (detailed scenarios)"
echo "  - TASK_11_DEPLOYMENT_CHECKLIST.md (progress tracking)"
echo ""

# Check if VS Code is available
if command -v code &> /dev/null; then
    code STREAMLIT_TESTING_GUIDE.md
    code TASK_11_DEPLOYMENT_CHECKLIST.md
    echo -e "${GREEN}✓ Opened testing guides in VS Code${NC}"
else
    echo "  Open these files manually in your editor"
fi
echo ""

# Step 4: Start Streamlit
echo -e "${YELLOW}Step 4: Starting Streamlit server...${NC}"
echo "  Server will start at: http://localhost:8501"
echo ""
echo "======================================================================"
echo "MANUAL TESTING REQUIRED"
echo "======================================================================"
echo ""
echo "Follow these scenarios in STREAMLIT_TESTING_GUIDE.md:"
echo ""
echo "  1. Scenario 1: Pure Education (Mode 1)"
echo "     → Ask: 'How do RAG systems work?'"
echo "     → Verify: ZERO resume mentions"
echo ""
echo "  2. Scenario 2: Hiring Signals (Mode 2)"
echo "     → Ask: 'We're hiring a Senior GenAI Engineer. How do RAG systems work?'"
echo "     → Verify: ONE subtle availability mention"
echo ""
echo "  3. Scenario 3: Explicit Request (Mode 3)"
echo "     → Ask: 'Can I get your resume?'"
echo "     → Provide email and name"
echo "     → Verify: Email received + SMS received"
echo ""
echo "  4. Scenario 4: Job Details Gathering"
echo "     → Continue from Scenario 3"
echo "     → Verify: Natural job details extraction"
echo ""
echo "  5. Scenario 5: Duplicate Prevention"
echo "     → Continue from Scenario 3"
echo "     → Ask for resume again"
echo "     → Verify: Polite rejection, no duplicate send"
echo ""
echo "  6. Scenario 6: Cross-Role Consistency"
echo "     → Switch to 'Software Developer' role"
echo "     → Verify: No resume distribution"
echo ""
echo "  7. Scenario 7: Invalid Email Detection"
echo "     → Provide invalid email"
echo "     → Verify: Graceful error handling"
echo ""
echo "======================================================================"
echo "After completing local tests, proceed to Vercel deployment:"
echo "  1. Review: TASK_11_DEPLOYMENT_CHECKLIST.md"
echo "  2. Deploy: vercel --prod"
echo "  3. Test production API with curl commands"
echo "======================================================================"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop the server when done testing${NC}"
echo ""

# Start Streamlit
streamlit run src/main.py
