import numpy as np
import time


class CrawlerService:
    @staticmethod
    def fetch_danmaku(bvid: str):
        """模拟：获取弹幕"""
        print(f"🕷️ [Crawler] 正在爬取 {bvid} 的弹幕...")
        # TODO: 这里填入 bilibili-api-python 的代码
        # text = sync(video.get_danmaku())...
        time.sleep(0.5)  # 模拟网络延迟
        return ["弹幕1: 泪目", "弹幕2: 太强了", "弹幕3: 这里的bgm好评"]


class ModelService:
    def __init__(self):
        # 这里加载你的 PyTorch/TensorFlow 模型
        print("🤖 [Model] 初始化情感分析模型...")
        pass

    def predict(self, danmaku_list):
        """输入弹幕列表，输出 128维 向量"""
        print(f"🧠 [Model] 正在计算情感向量 (输入 {len(danmaku_list)} 条弹幕)...")

        # TODO: 这里填入你的 Transformer 推理代码
        # inputs = tokenizer(danmaku_list, ...)
        # vector = model(inputs)...

        # 模拟生成归一化向量
        vec = np.random.rand(128)
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()


# 实例化模型服务 (避免每次请求都重新加载模型)
model_service = ModelService()