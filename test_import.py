try:
    from saas_endpoints import router
    print("SUCCESS: saas_endpoints imported")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
