#!/usr/bin/env python3
"""Serve native, pretrained and externally hosted models through one API.

Example:
  MINI_LLM_BACKEND=cupy python serve.py --registry ./serving_models.example.json
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", required=True, help="JSON model registry (local and remote providers)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--api-key", default=os.environ.get("MYLOCAL_LLM_API_KEY"))
    args = p.parse_args()
    from mini_llm.serving_api import ModelManager, create_server
    from mini_llm.serving_models import load_specs
    manager = ModelManager(load_specs(args.registry))
    server = create_server(manager, host=args.host, port=args.port, api_key=args.api_key)
    print(f"MyLocal_LLM OpenAI-style API at http://{args.host}:{args.port}/v1", flush=True)
    print(f"Models: {', '.join(manager.registry)} (lazy load, one local GPU resident)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
