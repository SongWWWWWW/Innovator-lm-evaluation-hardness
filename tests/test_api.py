
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="EMPTY", 
    base_url="http://61.175.246.233:8002/v1"
)

def chat_with_qwen():
    try:
        response = client.chat.completions.create(
            model="stage_2_instruct_Mixed_40M_1111_shuffled_v1_iter_0135000_hf",
            messages=[
                {"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "请解释一下什么是模型量化，并分步骤说明。"}
            ],
            stream=True,  # 建议开启流式输出，尤其是思维链模型（Thinking Model）
            extra_body={
                # 如果模型支持思维链输出，有时需要通过这个参数控制
                # "include_reasoning": True 
            }
        )

        print(f"助手回复：")
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

    except Exception as e:
        print(f"调用失败: {e}")

if __name__ == "__main__":
    chat_with_qwen()