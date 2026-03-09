import os
import subprocess
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
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


SYSTEM = f"""You are a coding agent at {os.getcwd()}.
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


TOOLS = [{
    "name": "bash",
    "description": "Run a shell command.",
    "input_schema": {
        "type": "object",
        "properties": {         # 定义 command 参数
            "command": { 
                "type": "string",
                "description": "The shell command to run"
            }
        },
        "required": ["command"] # 必须包含 command 参数
    },
}]

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

def agent_loop(messages: list):
    max_loops = 15 # 安全限制，防止 LLM 卡住时无限循环
    loop_count = 0
    last_command = None # 跟踪上一个命令以检测重复循环
    
    while loop_count < max_loops:
        loop_count += 1
        with client.messages.stream(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000
        ) as stream:
            # 遍历 text_stream，实现打字机效果的实时输出
            # end="" 防止了每次打印都换行
            # flush=True 会强制 Python 绕过系统缓冲区，立刻把接收到的字符推送到终端屏幕上
            for text in stream.text_stream:
                print(text, end="", flush=True)

            print()
            # 阻塞等待，直到获取完整的 Message 对象
            response = stream.get_final_message()

        messages.append({"role": "assistant", "content": response.content})
        # 如果不是工具调用，直接返回
        if response.stop_reason != "tool_use": 
            return
        
        results = []
        for block in response.content:
            if block.type == "tool_use":
                # LLM 有时会产生没有参数或 Schema 错误的工具调用。
                # 捕获 KeyError 防止 Agent 崩溃。
                try:
                    cmd = block.input['command']
                except KeyError:
                    error_msg = "Error: Missing 'command' parameter in tool call."
                    print(f"\033[31m[System] {error_msg}\033[0m")
                    results.append({
                        "type": "tool_result", 
                        "tool_use_id": block.id,
                        "content": error_msg
                    })
                    continue

                print(f"\033[33m$ {cmd}\033[0m")
                
                # 如果 LLM 重试刚刚失败（或成功）的完全相同的命令，
                # 它很可能陷入了循环。这里强制停止循环。
                if cmd == last_command:
                    error_msg = "Error: Duplicate command detected. Task stopped by system safety guard."
                    print(f"\033[31m[System] {error_msg}\033[0m")
                    results.append({
                        "type": "tool_result", 
                        "tool_use_id": block.id,
                        "content": error_msg
                    })
                    messages.append({"role": "user", "content": results})
                    return
                
                output = run_bash(cmd)
                last_command = cmd
                
                # 截断输出以避免长日志刷屏
                display_out = output[:100].replace('\n', ' ') + "..." if len(output) > 100 else output
                print(f"\033[90m{display_out}\033[0m")
                
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output
                })
        messages.append({"role": "user", "content": results})
        

if __name__ == "__main__":
    history = []
    while True:
        MODEL = choose_model(MODEL)
        try:
            query = input(f"\033[36ms01 ({MODEL}) >> \033[0m")
        except (EOFError, KeyboardInterrupt): # 捕获 Ctrl+D 和 Ctrl+C
            break
        if query.strip().lower() in ("exit", "q", ""):
            break
        
        history.append({"role": "user", "content": query})
        agent_loop(history)
        # response_content = history[-1]["content"] # 提取助手的回复
        # if isinstance(response_content, list): # 检查是否为列表
        #     for block in response_content:
        #         if hasattr(block, "text"):
        #             print(block.text)
        print()

