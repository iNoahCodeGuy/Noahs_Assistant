"""
Example: Using individual components of the modular system.
This shows the new capability to use components independently.
"""

from src.analytics.data_management import (
    PrivacyManager,
    DataQualityMonitor,
    BackupManager,
    PerformanceMonitor
)
import sqlite3


def component_usage_example():
    """Demonstrate using components independently."""

    print("🧩 Component-Specific Usage Example")
    print("=" * 50)

    # 1. Privacy Manager - Use independently
    print("\n🔒 Privacy Manager Example:")
    privacy = PrivacyManager()

    sensitive_text = "Contact John Doe at john.doe@company.com or call 555-123-4567"
    anonymized = privacy.anonymize_text(sensitive_text)
    print(f"Original: {sensitive_text}")
    print(f"Anonymized: {anonymized}")

    # Check for PII
    has_pii = privacy.check_pii_presence("My email is test@example.com")
    print(f"Contains PII: {has_pii}")

    # 2. Data Quality Monitor - Use with any database
    print("\n📊 Data Quality Monitor Example:")

    # Create a test database
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            response_time REAL,
            success BOOLEAN
        )
    """)

    # Insert test data
    test_records = [
        (1, "User 1", "user1@test.com", 2.5, True),
        (2, "User 2", None, 1.8, True),  # Missing email
        (3, "User 3", "user3@test.com", -1.0, False),  # Invalid response time
    ]
    cursor.executemany("INSERT INTO test_data VALUES (?, ?, ?, ?, ?)", test_records)
    conn.commit()

    # Use quality monitor
    quality = DataQualityMonitor(conn)

    # Note: This would need the actual user_interactions table for full functionality
    # This is just to show the component interface
    print("Quality monitor initialized successfully")

    # 3. Backup Manager - Use independently
    print("\n💾 Backup Manager Example:")
    backup_manager = BackupManager(":memory:")  # In-memory database for demo

    # Get backup status
    status = backup_manager.get_backup_status()
    print(f"Backup system status: {status['backup_count']} backups available")

    # 4. Performance Monitor - Use independently
    print("\n⚡ Performance Monitor Example:")
    perf_monitor = PerformanceMonitor(conn)

    # Configure custom alert
    perf_monitor.configure_alert(
        alert_name="custom_metric",
        threshold=5.0,
        operator="gt",
        enabled=True
    )

    alert_status = perf_monitor.get_alert_status()
    print(f"Configured alerts: {len(alert_status)}")

    conn.close()
    print("\n✅ Component examples completed!")


if __name__ == "__main__":
    component_usage_example()
