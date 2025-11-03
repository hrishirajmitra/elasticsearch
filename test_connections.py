#!/usr/bin/env python3
"""
Test Docker connections for PostgreSQL and Redis
"""
import sys

def test_postgresql():
    """Test PostgreSQL connection"""
    print("Testing PostgreSQL connection...")
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host='127.0.0.1',
            port=5432,
            dbname='searchdb',
            user='searchuser',
            password='searchpass'
        )
        
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"  ✓ PostgreSQL connected successfully!")
            print(f"  Version: {version[:50]}...")
        
        conn.close()
        return True
        
    except ImportError:
        print("  ✗ psycopg2 not installed")
        print("  Run: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("  Make sure PostgreSQL is running: docker-compose up -d postgres")
        return False


def test_redis():
    """Test Redis connection"""
    print("\nTesting Redis connection...")
    try:
        import redis
        
        client = redis.Redis(
            host='127.0.0.1',
            port=6379,
            db=0,
            decode_responses=True
        )
        
        # Test ping
        response = client.ping()
        if response:
            print(f"  ✓ Redis connected successfully!")
            
            # Test set/get
            client.set('test_key', 'test_value')
            value = client.get('test_key')
            print(f"  ✓ Set/Get test passed: {value}")
            client.delete('test_key')
            
            # Show Redis info
            info = client.info('server')
            print(f"  Version: {info['redis_version']}")
        
        client.close()
        return True
        
    except ImportError:
        print("  ✗ redis not installed")
        print("  Run: pip install redis")
        return False
    except Exception as e:
        print(f"  ✗ Connection failed: {e}")
        print("  Make sure Redis is running: docker-compose up -d redis")
        return False


def main():
    print("=" * 60)
    print("DOCKER SERVICES CONNECTION TEST")
    print("=" * 60)
    
    pg_ok = test_postgresql()
    redis_ok = test_redis()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PostgreSQL: {'✓ READY' if pg_ok else '✗ NOT READY'}")
    print(f"Redis:      {'✓ READY' if redis_ok else '✗ NOT READY'}")
    
    if pg_ok and redis_ok:
        print("\n✓ All services are ready!")
        print("You can now run: python plot_a.py")
        return 0
    else:
        print("\n✗ Some services are not ready")
        print("Fix the issues above before running plot_a.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())