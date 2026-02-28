#!/usr/bin/env python3
"""Generate test files for External Services setup.

This script creates placeholder files for testing storage uploads:
- test_resume.pdf: Sample resume PDF
- test_headshot.jpg: Sample profile image

These files are used by setup_external_services.py for integration testing.
"""

import os
from pathlib import Path

# Try to use reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("⚠️  reportlab not installed. Creating minimal PDF.")

# Try to use PIL for image generation
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("⚠️  Pillow not installed. Creating minimal image.")


def create_test_resume(filepath: Path):
    """Create a test resume PDF file."""
    if HAS_REPORTLAB:
        # Create a proper PDF with reportlab
        c = canvas.Canvas(str(filepath), pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(72, height - 72, "Noah Delacalzada")
        
        # Contact info
        c.setFont("Helvetica", 12)
        c.drawString(72, height - 100, "Email: noah@example.com | Phone: (555) 123-4567")
        c.drawString(72, height - 115, "Portfolio: noahdelacalzada.com | GitHub: github.com/noah")
        
        # Section: Summary
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 160, "Professional Summary")
        c.setFont("Helvetica", 11)
        summary = [
            "Full-stack developer with expertise in AI/ML, cloud architecture, and data engineering.",
            "Passionate about building intelligent systems that solve real-world problems.",
            "Strong background in Python, TypeScript, PostgreSQL, and modern web frameworks."
        ]
        y_position = height - 185
        for line in summary:
            c.drawString(90, y_position, line)
            y_position -= 15
        
        # Section: Experience
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 260, "Experience")
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(90, height - 285, "Senior Software Engineer | Tech Company")
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(90, height - 300, "Jan 2023 - Present")
        c.setFont("Helvetica", 11)
        experience = [
            "• Built AI-powered assistant platform using OpenAI, LangChain, and Supabase",
            "• Designed and implemented vector search with pgvector for knowledge retrieval",
            "• Developed RESTful APIs with Next.js and deployed on Vercel",
            "• Integrated multi-channel notifications (email, SMS) for real-time alerts"
        ]
        y_position = height - 320
        for line in experience:
            c.drawString(90, y_position, line)
            y_position -= 15
        
        # Section: Skills
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, height - 420, "Technical Skills")
        c.setFont("Helvetica", 11)
        skills = [
            "• Languages: Python, TypeScript, JavaScript, SQL",
            "• Frameworks: Next.js, React, FastAPI, LangChain",
            "• Cloud: Supabase, Vercel, AWS, GCP",
            "• Databases: PostgreSQL, MongoDB, Redis, SQLite",
            "• AI/ML: OpenAI, pgvector, RAG, embeddings"
        ]
        y_position = height - 445
        for line in skills:
            c.drawString(90, y_position, line)
            y_position -= 15
        
        # Footer
        c.setFont("Helvetica-Oblique", 9)
        c.drawString(72, 50, "Generated for External Services testing - Noah's AI Assistant")
        
        c.save()
        print(f"✅ Created test resume: {filepath}")
    else:
        # Create minimal PDF without reportlab
        # PDF header
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
>>
endobj

4 0 obj
<<
/Length 150
>>
stream
BT
/F1 24 Tf
72 720 Td
(Noah Delacalzada - Test Resume) Tj
0 -30 Td
/F1 12 Tf
(This is a test resume for External Services setup) Tj
0 -20 Td
(Email: noah@example.com) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000317 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
516
%%EOF
"""
        
        with open(filepath, 'wb') as f:
            f.write(pdf_content)
        print(f"✅ Created minimal test resume: {filepath}")


def create_test_headshot(filepath: Path):
    """Create a test headshot image."""
    if HAS_PIL:
        # Create a proper image with PIL
        # Create a gradient background
        width, height = 400, 400
        img = Image.new('RGB', (width, height), color='#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        # Draw gradient background
        for i in range(height):
            color_value = int(26 + (i / height) * 40)
            draw.rectangle([(0, i), (width, i + 1)], fill=(color_value, color_value, color_value + 20))
        
        # Draw a circle for the "face"
        circle_center = (width // 2, height // 2)
        circle_radius = 120
        draw.ellipse(
            [circle_center[0] - circle_radius, circle_center[1] - circle_radius,
             circle_center[0] + circle_radius, circle_center[1] + circle_radius],
            fill='#16213e',
            outline='#0f3460',
            width=3
        )
        
        # Draw "N" for Noah
        try:
            # Try to use a truetype font
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 140)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        # Draw the letter
        text = "N"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_position = (
            circle_center[0] - text_width // 2,
            circle_center[1] - text_height // 2 - 10
        )
        draw.text(text_position, text, fill='#00d4ff', font=font)
        
        # Add label at bottom
        try:
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            label_font = ImageFont.load_default()
        
        label_text = "Noah Delacalzada"
        bbox = draw.textbbox((0, 0), label_text, font=label_font)
        label_width = bbox[2] - bbox[0]
        label_position = (width // 2 - label_width // 2, height - 40)
        draw.text(label_position, label_text, fill='#00d4ff', font=label_font)
        
        # Save image
        img.save(filepath, 'JPEG', quality=90)
        print(f"✅ Created test headshot: {filepath}")
    else:
        # Create minimal JPEG without PIL
        # Minimal JPEG header (8x8 gray square)
        jpeg_data = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
            0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
            0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
            0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
            0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
            0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
            0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
            0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08,
            0x00, 0x08, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x03, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00,
            0x3F, 0x00, 0x00, 0xFF, 0xD9
        ])
        
        with open(filepath, 'wb') as f:
            f.write(jpeg_data)
        print(f"✅ Created minimal test headshot: {filepath}")


def main():
    """Generate test files for External Services setup."""
    print("📁 Generating test files for External Services setup...")
    print()
    
    # Ensure data directory exists
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Create test files
    resume_path = data_dir / 'test_resume.pdf'
    headshot_path = data_dir / 'test_headshot.jpg'
    
    create_test_resume(resume_path)
    create_test_headshot(headshot_path)
    
    print()
    print("✅ Test files created successfully!")
    print()
    print("Files created:")
    print(f"  • {resume_path} ({resume_path.stat().st_size} bytes)")
    print(f"  • {headshot_path} ({headshot_path.stat().st_size} bytes)")
    print()
    print("Next step: Run the External Services setup script")
    print("  python scripts/setup_external_services.py")


if __name__ == "__main__":
    main()
