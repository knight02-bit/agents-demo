import os
import subprocess
import json
from anthropic import Anthropic
from langfuse import observe, get_client

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# 全局初始化 client, 并检查认证
langfuse = get_client()
try:
    ok = langfuse.auth_check()
except Exception as e:
    ok = False
    print(f"Langfuse auth_check failed: {e}")

if ok:
    print("Langfuse client is authenticated and ready!")
else:
    print("Langfuse authentication failed. Please set LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY and (optionally) LANGFUSE_HOST.")

if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)


client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
WORKDIR = Path(os.getcwd())
COMMON_MODELS = [
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
    "gpt-3.5-turbo",
    "deepseek-chat",
    "deepseek-r1",
    "glm-4.5",
    "qwen-plus",
    "kimi-k2.5",
    "MiniMax-M2.1",
]
MODEL = os.getenv("MODEL_ID")

def choose_model(current_model: str) -> str:
    """选择模型"""
    print("\033[35m选择模型：\033[0m")
    for idx, name in enumerate(COMMON_MODELS, start=1):
        print(f"{idx}. {name}")
    print(f"{len(COMMON_MODELS)+1}. 自定义输入")
    choice = input(f"\033[35m模型编号（回车沿用: {current_model}）>> \033[0m").strip()
    if not choice:
        return current_model

    if choice.isdigit():
        index = int(choice)
        if 1 <= index <= len(COMMON_MODELS):
            return COMMON_MODELS[index-1]
        elif index == len(COMMON_MODELS) + 1:
            return input("\033[35m请输入自定义模型ID>> \033[0m").strip()
    print("\033[31m无效输入，请输入模型编号\033[0m")
    return choose_model(current_model)


SYSTEM = f"""You are a coding agent at {WORKDIR}.
Platform: {os.name} (Windows).
Shell: cmd.exe.

Instructions:
1. Use the available 'bash' tool to execute commands.
2. IMPORTANT: This is Windows cmd.exe. 'mkdir' does NOT support '-p'. Use 'mkdir <path>'.
3. If a command returns "(no output)", it usually means success.
4. If you see "already exists" or "已存在", consider the task DONE. Do NOT retry.
5. When the task is complete, verify if needed, then respond with a short text summary to finish.
6. DO NOT continue calling tools if the task is done.
"""

def safe_path(path: str) -> Path:
    """路径沙箱防止逃逸工作区"""
    path = (WORKDIR / path).resolve() # 确保路径是绝对路径
    if not path.is_relative_to(WORKDIR): # 确保路径在工作区内
        return f"Error: Path {path} is outside of working directory {WORKDIR}"
    return path


def run_read(path:str, limit: int = None) -> str:
    """读取文件内容，可选限制行数"""
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and len(lines) > limit:
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000] # 限制输出长度
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """写入文件内容"""
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True) # 创建父目录（如果不存在）
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """编辑文件内容，替换 old_text 为 new_text"""
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: {old_text} not found in {path}"
        fp.write_text(content.replace(old_text, new_text)) # 替换所有匹配项
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 针对 Windows 的健壮编码处理
        # text=False 捕获原始字节，允许手动尝试解码。
        # 这修复了 Windows cmd.exe 输出 GBK (CP936) 但 Python 期望 UTF-8 的问题，
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=False, timeout=120)
        
        out_bytes = r.stdout + r.stderr
        try:
            out = out_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                out = out_bytes.decode('gbk') # 针对中文 Windows 回退到 GBK
            except UnicodeDecodeError:
                out = out_bytes.decode('utf-8', errors='replace')
                
        out = out.strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

# dispatch map 将工具名映射到处理函数
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": { 
                    "type": "string",
                    "description": "The shell command to run"
                }
            },
            "required": ["command"] # 必须包含 command 参数
        }
    },
    {
        "name": "read_file",
        "description": "Read file contents.",
        "input_schema":{
            "type": "object", # 
            "properties":{
                "path":{
                    "type": "string",
                    "description": "The path to the file to read"
                },
                "limit":{
                    "type": "integer",
                    "description": "Optional limit on number of lines to read (default 100)"
                }
            },
            "required": ["path"] # 必须包含 path 参数
        }
    },
    {
        "name": "write_file",
        "description": "Write file contents.",
        "input_schema":{
            "type": "object", # 
            "properties":{
                "path":{
                    "type": "string",
                    "description": "The path to the file to write"
                },
                "content":{
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["path", "content"] # 必须包含 path 和 content 参数
        }
    },
    {
        "name": "edit_file",
        "description": "Replace exact text in file.",
        "input_schema":{
            "type": "object",
            "properties":{
                "path":{
                    "type": "string",
                    "description": "The path to the file to edit"
                },
                "old_text":{
                    "type": "string",
                    "description": "The text to replace"
                },
                "new_text":{
                    "type": "string",
                    "description": "The text to replace with"
                }
            },
            "required": ["path", "old_text", "new_text"] # 必须包含 path, old_text, new_text 参数
        }
    }
]



@observe()
def agent_loop(messages: list):
    # 初始化 Langfuse 跟踪
    langfuse.update_current_trace(
        user_id="user_123",
        session_id="session_abc",
        tags=[MODEL], # 推荐用 tags 记录模型
        version="1.0.0"
    )

    max_loops = 15 # 安全限制，防止 LLM 卡住时无限循环
    loop_count = 0
    last_tool_call = None # 跟踪上一个工具调用（tool + input）以检测重复循环

    while loop_count < max_loops:
        loop_count += 1

        # 为每次大模型调用创建一个独立的 Generation 节点
        # 这会让 Langfuse 后台呈现极其清晰的瀑布流并精准挂载 Token
        with langfuse.start_as_current_observation(
            name=f"llm-call-loop-{loop_count}",
            as_type="generation",
            model=MODEL,
            input=messages # 可选：记录丢给大模型的当前完整上下文
        ) as generation:
            with client.messages.stream(
                model=MODEL, system=SYSTEM, messages=messages,
                tools=TOOLS, max_tokens=5000
            ) as stream:
                # 遍历 text_stream，实现打字机效果的实时输出
                # end="" 防止了每次打印都换行
                # flush=True 会强制 Python 绕过系统缓冲区，立刻把接收到的字符推送到终端屏幕上
                for text in stream.text_stream:
                    print(text, end="", flush=True)

                print()
                # 阻塞等待，直到获取完整的 Message 对象
                response = stream.get_final_message()
                print(f"Token 消耗: {response.usage}")

        # 🌟 将当前请求的 Token 消耗和输出更新给当前的 Generation
        if hasattr(response, 'usage') and response.usage:
            generation.update(
                usage={
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens
                }
            )

        messages.append({"role": "assistant", "content": response.content}) # 把大模型的回复添加到上下文
        # 如果不是工具调用，直接返回
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                # LLM 有时会产生没有参数或 Schema 错误的工具调用。
                # 捕获 KeyError 防止 Agent 崩溃。
                try:
                    tool_call_sig = json.dumps( # 对工具调用参数进行排序，确保重复调用可检测
                        {"tool": block.name, "input": block.input},
                        ensure_ascii=False, # 确保非 ASCII 字符正常显示
                        sort_keys=True, # 按键排序，确保重复调用可检测
                    )

                    # 如果 LLM 重试刚刚失败（或成功）的完全相同的工具调用，很可能陷入了循环
                    if tool_call_sig == last_tool_call:
                        warning_msg = f"Warning: Duplicate tool call detected ({block.name}). Task may be stuck in a loop."
                        print(f"\033[33m[System] {warning_msg}\033[0m")
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": warning_msg
                        })
                        messages.append({"role": "user", "content": results})
                        return                                                                                                                                                                                                                                                                                      

                    last_tool_call = tool_call_sig

                    handler = TOOL_HANDLERS.get(block.name) # 获取工具处理函数
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"

                    # 记录工具调用结果到 Langfuse
                    generation.update(
                        name=f"tool-call-{block.name}",
                        as_type="tool",
                        input=block.input,
                        output=output
                    )
                    
                    display_out = output[:200].replace('\n', ' ') + "..." if len(output) > 200 else output
                    print(f"\033[90m[Tool] {block.name}: {display_out}\033[0m")

                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output
                    })
                except KeyError:
                    error_msg = "Error: Missing required parameter in tool call."
                    print(f"\033[31m[System] {error_msg}\033[0m")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": error_msg
                    })
                    continue
        messages.append({"role": "user", "content": results}) # 把工具调用结果添加到上下文
        

if __name__ == "__main__":
    history = []
    while True:
        MODEL = choose_model(MODEL)
        try:
            query = input(f"\033[36ms02 ({MODEL}) >> \033[0m")
        except (EOFError, KeyboardInterrupt): # 捕获 Ctrl+D 和 Ctrl+C
            break
        if query.strip().lower() in ("exit", "q", ""):
            break
        
        history.append({"role": "user", "content": query})
        agent_loop(history)
        print()

    # 确保所有事件都被发送到 Langfuse
    langfuse.flush()
