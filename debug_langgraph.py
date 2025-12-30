import sys
import traceback

try:
    from langgraph_cli.cli import dev
    sys.argv = ['dev', '--port', '2024']
    dev()
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
