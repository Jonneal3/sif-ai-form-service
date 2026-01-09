#!/usr/bin/env python3
"""
Quick test script to verify Supabase connection.
"""

import os
import sys
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv not installed, try manual load
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

from api.supabase_client import get_supabase_client

def test_connection():
    """Test Supabase connection."""
    print("🔌 Testing Supabase connection...")
    print()
    
    # Check env vars
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    
    print(f"SUPABASE_URL: {url[:50]}..." if url else "❌ SUPABASE_URL not set")
    print(f"SUPABASE_SERVICE_ROLE_KEY: {'✅ Set' if key else '❌ Not set'}")
    print()
    
    if not url or not key:
        print("❌ Missing required environment variables!")
        return False
    
    # Get client
    client = get_supabase_client()
    if not client:
        print("❌ Failed to create Supabase client")
        return False
    
    print("✅ Supabase client created successfully")
    print()
    
    # Test a simple query (list tables or get a count)
    try:
        print("🧪 Testing database query...")
        
        # Try to query a common table (adjust table name as needed)
        # This will fail if table doesn't exist, but that's okay - we're just testing connection
        result = client.table("sessions").select("id", count="exact").limit(1).execute()
        
        print(f"✅ Query successful! Found {result.count if hasattr(result, 'count') else 'N/A'} records")
        return True
        
    except Exception as e:
        # If table doesn't exist, that's fine - connection works
        error_msg = str(e)
        if "relation" in error_msg.lower() or "does not exist" in error_msg.lower():
            print(f"✅ Connection works! (table 'sessions' doesn't exist yet - create it in Supabase)")
            print()
            print("📝 Next steps:")
            print("   1. Create your tables in Supabase Dashboard")
            print("   2. Update table/column names in api/supabase_client.py to match your schema")
            return True
        else:
            print(f"❌ Query failed: {error_msg}")
            return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

