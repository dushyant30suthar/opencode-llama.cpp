#!/usr/bin/env python3
"""
NVIDIA ships its Llama-Nemotron tool parser as a plugin file inside the
checkpoint, written against vllm==0.9.2 (the version its model card pins).
vLLM 0.26 reorganised the modules it imports, so it fails at load with
ModuleNotFoundError and the parser never registers:

    KeyError: 'invalid tool call parser: llama_nemotron_json'

The parser LOGIC is fine — only the import block is stale. Patch a copy rather
than the checkpoint, so the original stays pristine and a re-download does not
silently revert the fix.

Verified against this venv:
  vllm.entrypoints.openai.protocol            -> gone; split into
      .engine.protocol            ToolCall, FunctionCall, Delta{Tool,Function}Call
      abstract_tool_parser re-exports ChatCompletionRequest, DeltaMessage,
                                      ExtractedToolCallInformation
  vllm.entrypoints.openai.tool_parsers.*      -> vllm.tool_parsers.*
  vllm.transformers_utils.tokenizer.AnyTokenizer -> vllm.tokenizers.TokenizerLike
  vllm.utils.random_uuid                      -> unchanged
"""
import pathlib
import shutil

SRC = pathlib.Path.home() / (
    ".lmstudio/models/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4/"
    "llama_nemotron_toolcall_parser_no_streaming.py"
)
DST_DIR = pathlib.Path.home() / "Projects/vllm/parsers"
DST = DST_DIR / "llama_nemotron_toolcall_parser.py"

OLD = """from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall, DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid"""

NEW = """# --- PATCHED FOR vLLM 0.26 (upstream file targets vllm==0.9.2) ---
# Only the imports changed; the parsing logic below is NVIDIA's, untouched.
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from vllm.tool_parsers.abstract_tool_parser import (
    ChatCompletionRequest,
    DeltaMessage,
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)
from vllm.logger import init_logger
# AnyTokenizer was renamed; aliased so the type hints below still resolve.
from vllm.tokenizers import TokenizerLike as AnyTokenizer
from vllm.utils import random_uuid
# --- END PATCH ---"""

DST_DIR.mkdir(parents=True, exist_ok=True)
text = SRC.read_text()
if OLD not in text:
    if "PATCHED FOR vLLM 0.26" in DST.read_text() if DST.exists() else False:
        print("already patched")
        raise SystemExit(0)
    raise SystemExit("import block not found verbatim — inspect the file by hand")

DST.write_text(text.replace(OLD, NEW, 1))
shutil.copystat(SRC, DST)
print(f"patched copy written: {DST}")
print("registers:", [
    l.split('"')[1] for l in DST.read_text().splitlines() if "register_module" in l
])
