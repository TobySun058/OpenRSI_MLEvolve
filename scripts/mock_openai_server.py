from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, payload, code=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(length) or b"{}")
        tools = data.get("tools") or []
        messages = data.get("messages", [])
        text = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
        last = str(messages[-1].get("content", "")) if messages else ""

        if tools:
            name = tools[0]["function"]["name"]
            if name == "determine_metric_direction":
                args = {"lower_is_better": False, "reasoning": "Accuracy is maximized."}
            elif name == "submit_code_review":
                args = {"needs_revision": False, "reasoning": "Looks good."}
            elif name == "submit_review":
                args = {"is_bug": False, "summary": "Ran successfully.", "metric": 0.42, "lower_is_better": False}
            elif name == "check_data_leakage":
                args = {"has_leakage": False, "leakage_reason": "No leakage.", "confidence": "low"}
            else:
                args = {"ok": True}
            return self._send({
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": data.get("model", "mock"),
                "choices": [{
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        if "Return only one JSON object" in text:
            content = '{"ok": true, "value": 7}'
        elif "Return a short plan" in text or "Improve the code" in text:
            content = (
                "Short plan followed by code.\n"
                "```python\n"
                "import pandas as pd\n"
                "pd.DataFrame({\"id\":[5,6], \"prediction\":[0.1,0.2]}).to_csv(\"submission/submission.csv\", index=False)\n"
                "print(\"Final Validation Score: 0.42\")\n"
                "```"
            )
        else:
            content = "FRONTIS_OK"

        return self._send({
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": data.get("model", "mock"),
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8002), Handler).serve_forever()
