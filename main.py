import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI #type: ignore
from fastapi.middleware.cors import CORSMiddleware #type: ignore
from pydantic import BaseModel, Field #type: ignore
from openai import OpenAI #type: ignore


app = FastAPI(title="Code Interpreter with AI Error Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int] = Field(
        description="One-based source line numbers responsible for the error"
    )


def execute_python_code(code: str) -> dict:
    """Execute code and capture stdout or the complete traceback."""
    old_stdout = sys.stdout
    captured = StringIO()
    sys.stdout = captured

    try:
        exec(compile(code, "<string>", "exec"), {})
        return {"success": True, "output": captured.getvalue()}

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc(),
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """Ask AI Pipe to return structured error line numbers."""
    token = os.environ.get("AIPIPE_TOKEN")
    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is missing")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openai/v1",
    )

    response = client.beta.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identify the one-based line number or numbers in the "
                    "provided Python source responsible for the exception. "
                    "Use the traceback as evidence. Return an empty list "
                    "only if no source line can be identified. Do not "
                    "confuse the final traceback line with the source line."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Python source code:\n{code}\n\n"
                    f"Traceback:\n{error_traceback}"
                ),
            },
        ],
        response_format=ErrorAnalysis,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("AI did not return structured error analysis")

    # Keep only valid one-based line numbers within the submitted source.
    return sorted({
        line
        for line in parsed.error_lines
        if 1 <= line <= len(code.splitlines())
    })


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"],
        )
    except Exception as e:
        print("AI analysis failed:", repr(e))
        error_lines = []

    return {
        "error": error_lines,
        "result": execution["output"],
    }


@app.get("/")
def health_check():
    return {"status": "ok", "endpoint": "/code-interpreter"}
