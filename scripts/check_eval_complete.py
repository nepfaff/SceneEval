#!/usr/bin/env python3
"""Check if a scene evaluation is complete (all metrics have results).

Usage: python3 check_eval_complete.py <eval_result.json>

Exit codes:
  0 - All metrics in "metrics" array have corresponding entries in "results"
  1 - Missing metrics or error reading file
"""
import json
import sys
import os

def main():
    if len(sys.argv) < 2:
        os._exit(1)
    
    try:
        with open(sys.argv[1]) as f:
            d = json.load(f)
        metrics = d.get('metrics', [])
        results = d.get('results', {})
        os._exit(0 if all(m in results for m in metrics) else 1)
    except:
        os._exit(1)

if __name__ == '__main__':
    main()
