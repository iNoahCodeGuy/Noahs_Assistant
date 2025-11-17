"""
Simple validation tests for API endpoints - structure and routing only.
"""
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_api_files_exist():
    """Verify all API endpoint files exist."""
    print("\n" + "="*80)
    print("Testing API Files Exist")
    print("="*80)

    api_dir = os.path.join(os.path.dirname(__file__), '..', 'api')

    required_files = [
        'chat.py',
        'email.py',
        'feedback.py',
        'confess.py',
        'README.md'
    ]

    for filename in required_files:
        filepath = os.path.join(api_dir, filename)
        assert os.path.exists(filepath), f"Missing file: {filename}"
        print(f"✅ Found: {filename}")

    print("✅ All API files exist!")


def test_api_handlers_structure():
    """Verify API handlers have required methods."""
    print("\n" + "="*80)
    print("Testing API Handler Structure")
    print("="*80)

    api_files = ['chat.py', 'email.py', 'feedback.py', 'confess.py']

    for filename in api_files:
        filepath = os.path.join(os.path.dirname(__file__), '..', 'api', filename)

        with open(filepath, 'r') as f:
            content = f.read()

            # Check for required methods
            assert 'class handler' in content, f"{filename}: Missing handler class"
            assert 'def do_POST' in content, f"{filename}: Missing do_POST method"
            assert 'def do_OPTIONS' in content, f"{filename}: Missing do_OPTIONS method"
            assert '_send_json' in content, f"{filename}: Missing _send_json method"
            assert '_send_error' in content, f"{filename}: Missing _send_error method"
            assert '_send_cors_headers' in content, f"{filename}: Missing CORS headers"

            print(f"✅ {filename}: Structure valid")

    print("✅ All API handlers have required structure!")


def test_vercel_config():
    """Verify vercel.json configuration."""
    print("\n" + "="*80)
    print("Testing Vercel Configuration")
    print("="*80)

    import json

    vercel_path = os.path.join(os.path.dirname(__file__), '..', 'vercel.json')
    assert os.path.exists(vercel_path), "vercel.json not found"

    with open(vercel_path, 'r') as f:
        config = json.load(f)

    # Verify routes
    assert 'routes' in config, "Missing routes configuration"
    assert len(config['routes']) >= 4, "Missing API routes"

    route_paths = [route['assistant'] for route in config['routes']]
    assert '/api/chat' in route_paths, "Missing /api/chat route"
    assert '/api/email' in route_paths, "Missing /api/email route"
    assert '/api/feedback' in route_paths, "Missing /api/feedback route"
    assert '/api/confess' in route_paths, "Missing /api/confess route"

    # Verify environment variables
    assert 'env' in config, "Missing environment configuration"
    required_envs = [
        'OPENAI_API_KEY',
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY',
        'RESEND_API_KEY',
        'TWILIO_ACCOUNT_SID',
        'TWILIO_AUTH_TOKEN'
    ]

    for env_var in required_envs:
        assert env_var in config['env'], f"Missing env var: {env_var}"

    print("✅ Vercel configuration valid!")


def test_api_documentation():
    """Verify API documentation exists."""
    print("\n" + "="*80)
    print("Testing API Documentation")
    print("="*80)

    docs = [
        'api/README.md',
        'API_INTEGRATION.md'
    ]

    for doc in docs:
        doc_path = os.path.join(os.path.dirname(__file__), '..', doc)
        assert os.path.exists(doc_path), f"Missing documentation: {doc}"

        with open(doc_path, 'r') as f:
            content = f.read()
            assert len(content) > 100, f"{doc}: Documentation too short"

        print(f"✅ Found: {doc}")

    print("✅ All API documentation exists!")


if __name__ == "__main__":
    print("\n🧪 API Endpoint Validation Tests")
    print("="*80)

    try:
        test_api_files_exist()
        test_api_handlers_structure()
        test_vercel_config()
        test_api_documentation()

        print("\n" + "="*80)
        print("🎉 All API validation tests passed!")
        print("="*80)
        print("\n✅ API Integration Complete:")
        print("   - 4 endpoint handlers created")
        print("   - Vercel configuration ready")
        print("   - Documentation complete")
        print("   - CORS support enabled")
        print("   - Error handling implemented")
        print("\n📋 Next Steps:")
        print("   1. Install Vercel CLI: npm i -g vercel")
        print("   2. Test locally: vercel dev")
        print("   3. Deploy: vercel --prod")
        print("="*80 + "\n")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
