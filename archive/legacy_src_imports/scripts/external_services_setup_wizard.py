#!/usr/bin/env python3
"""External Services Setup Master Script

This script guides you through the complete External Services setup process:
1. Check prerequisites (Supabase setup)
2. Generate test files
3. Configure services (interactive)
4. Run setup verification
5. Test integrations

Run this script to get started with External Services setup.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")


def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.OKCYAN}ℹ️  {text}{Colors.ENDC}")


def check_prerequisites():
    """Check if Supabase setup are completed."""
    print_header("Step 1: Checking Prerequisites")

    # Load environment variables
    load_dotenv()

    # Check for Supabase configuration
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

    if not supabase_url or supabase_url == 'your_supabase_project_url_here':
        print_error("Supabase not configured")
        print_info("You need to complete Supabase setup first")
        print_info("Please follow the setup guide:")
        print(f"  {Colors.BOLD}docs/PHASE_1_SETUP.md{Colors.ENDC}")
        print(f"  {Colors.BOLD}docs/PHASE_2_COMPLETE.md{Colors.ENDC}")
        return False

    if not supabase_key or supabase_key == 'your_service_role_key_here':
        print_error("Supabase service role key not configured")
        print_info("Add your Supabase credentials to .env file")
        return False

    print_success("Supabase configured")

    # Check for required packages
    try:
        import supabase
        print_success("supabase package installed")
    except ImportError:
        print_error("supabase package not installed")
        print_info("Run: pip install -r requirements.txt")
        return False

    return True


def check_service_configuration():
    """Check which services are configured."""
    print_header("Step 2: Checking Service Configuration")

    load_dotenv()

    services = {
        'Resend Email': {
            'key': 'RESEND_API_KEY',
            'required_vars': ['RESEND_API_KEY', 'RESEND_FROM_EMAIL', 'ADMIN_EMAIL'],
            'signup_url': 'https://resend.com',
            'docs': 'https://resend.com/docs'
        },
        'Twilio SMS': {
            'key': 'TWILIO_ACCOUNT_SID',
            'required_vars': ['TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN',
                             'TWILIO_PHONE_NUMBER', 'ADMIN_PHONE_NUMBER'],
            'signup_url': 'https://www.twilio.com/try-twilio',
            'docs': 'https://www.twilio.com/docs/sms'
        }
    }

    configured_services = []
    missing_services = []

    for service_name, service_info in services.items():
        key_value = os.getenv(service_info['key'])

        if key_value and not key_value.startswith('your_') and not key_value.startswith('re_your') and key_value != 'ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx':
            # Check all required vars
            all_configured = True
            for var in service_info['required_vars']:
                value = os.getenv(var)
                if not value or 'your_' in value or 'xxxxxx' in value:
                    all_configured = False
                    break

            if all_configured:
                print_success(f"{service_name} configured")
                configured_services.append(service_name)
            else:
                print_warning(f"{service_name} partially configured (some variables missing)")
                missing_services.append((service_name, service_info))
        else:
            print_warning(f"{service_name} not configured")
            missing_services.append((service_name, service_info))

    return configured_services, missing_services


def show_setup_instructions(missing_services):
    """Show setup instructions for missing services."""
    if not missing_services:
        return

    print_header("Step 3: Service Setup Instructions")

    print("The following services need to be configured:\n")

    for service_name, service_info in missing_services:
        print(f"{Colors.BOLD}{service_name}:{Colors.ENDC}")
        print(f"  1. Sign up: {Colors.OKBLUE}{service_info['signup_url']}{Colors.ENDC}")
        print(f"  2. Get API credentials")
        print(f"  3. Add to .env file")
        print(f"  4. See docs: {Colors.OKBLUE}{service_info['docs']}{Colors.ENDC}")
        print()

    print(f"{Colors.BOLD}Detailed setup guide:{Colors.ENDC}")
    print(f"  {Colors.OKCYAN}docs/EXTERNAL_SERVICES_SETUP_GUIDE.md{Colors.ENDC}")
    print()


def generate_test_files():
    """Generate test files for upload testing."""
    print_header("Step 4: Generating Test Files")

    try:
        # Import and run the generate_test_files script
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))

        from generate_test_files import create_test_resume, create_test_headshot

        data_dir = Path(__file__).parent.parent / 'data'
        data_dir.mkdir(exist_ok=True)

        resume_path = data_dir / 'test_resume.pdf'
        headshot_path = data_dir / 'test_headshot.jpg'

        if not resume_path.exists():
            create_test_resume(resume_path)
        else:
            print_info(f"Test resume already exists: {resume_path}")

        if not headshot_path.exists():
            create_test_headshot(headshot_path)
        else:
            print_info(f"Test headshot already exists: {headshot_path}")

        print_success("Test files ready")
        return True
    except Exception as e:
        print_error(f"Failed to generate test files: {e}")
        print_info("You can manually create test files in the data/ directory")
        return False


def run_setup_script(configured_services):
    """Run the main External Services setup script."""
    print_header("Step 5: Running Setup Script")

    if not configured_services:
        print_warning("No services configured yet")
        print_info("Please configure at least one service before running setup")
        print_info("See: docs/EXTERNAL_SERVICES_SETUP_GUIDE.md")
        return False

    print_info("Running External Services setup script...")
    print_info("This will:")
    print("  • Create storage buckets")
    print("  • Upload test files")
    print("  • Test configured services")
    print()

    # Ask for confirmation
    response = input(f"{Colors.BOLD}Continue? [Y/n]: {Colors.ENDC}").strip().lower()
    if response and response != 'y' and response != 'yes':
        print_info("Setup cancelled")
        return False

    # Run the setup script
    script_dir = Path(__file__).parent
    setup_script = script_dir / 'setup_external_services.py'

    if setup_script.exists():
        print()
        os.system(f"python {setup_script}")
        return True
    else:
        print_error(f"Setup script not found: {setup_script}")
        return False


def show_next_steps():
    """Show next steps after setup."""
    print_header("Next Steps")

    print(f"{Colors.BOLD}1. Test the integration example:{Colors.ENDC}")
    print(f"   cd examples")
    print(f"   python contact_form_integration.py")
    print()

    print(f"{Colors.BOLD}2. Read the documentation:{Colors.ENDC}")
    print(f"   • {Colors.OKCYAN}docs/EXTERNAL_SERVICES_COMPLETE.md{Colors.ENDC} - Comprehensive guide")
    print(f"   • {Colors.OKCYAN}docs/EXTERNAL_SERVICES_QUICK_REFERENCE.md{Colors.ENDC} - Quick reference")
    print()

    print(f"{Colors.BOLD}3. Start using the services:{Colors.ENDC}")
    print(f"   from src.services import get_storage_service, get_resend_service, get_twilio_service")
    print()

    print(f"{Colors.BOLD}4. Move to Phase 4:{Colors.ENDC}")
    print(f"   • Create Next.js API routes")
    print(f"   • Deploy to Vercel")
    print(f"   • Configure production environment")
    print()


def main():
    """Main setup workflow."""
    print_header("🚀 External Services Setup - Storage, Email & SMS")

    print(f"{Colors.BOLD}This wizard will guide you through:{Colors.ENDC}")
    print("  ✓ Checking prerequisites")
    print("  ✓ Configuring external services")
    print("  ✓ Setting up storage buckets")
    print("  ✓ Testing integrations")
    print()

    # Step 1: Check prerequisites
    if not check_prerequisites():
        print()
        print_error("Prerequisites not met. Please complete Supabase setup first.")
        sys.exit(1)

    # Step 2: Check service configuration
    configured_services, missing_services = check_service_configuration()

    # Step 3: Show setup instructions if needed
    if missing_services:
        show_setup_instructions(missing_services)

        print(f"{Colors.BOLD}Would you like to configure services now?{Colors.ENDC}")
        print("  • Open .env file and add your API credentials")
        print("  • See docs/EXTERNAL_SERVICES_SETUP_GUIDE.md for detailed instructions")
        print()
        response = input(f"{Colors.BOLD}Continue after configuring? [Y/n]: {Colors.ENDC}").strip().lower()

        if response and response != 'y' and response != 'yes':
            print_info("Setup paused. Run this script again when ready.")
            sys.exit(0)

        # Reload environment and check again
        load_dotenv(override=True)
        configured_services, missing_services = check_service_configuration()

        if not configured_services:
            print_warning("No services configured yet")
            print_info("You can continue with just storage, or configure services first")

    # Step 4: Generate test files
    generate_test_files()

    # Step 5: Run setup script
    if configured_services or True:  # Always allow storage setup
        run_setup_script(configured_services)

    # Show next steps
    show_next_steps()

    print_success("External Services setup wizard complete!")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_info("Setup cancelled by user")
        sys.exit(0)
