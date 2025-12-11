from .database import db
from .services import CrawlerService, model_service


class RecommendationPipeline:
    def __init__(self, bvid):
        self.bvid = bvid
        # 上下文：用于在管道各阶段传递数据
        self.context = {
            "bvid": bvid,
            "vector": None,  # 核心中间态：情感向量
            "recommendations": []  # 最终结果
        }

    def run(self):
        """执行管道流"""
        # 步骤 1: 检查缓存
        self._step_check_database()

        # 步骤 2: (如果没缓存) 获取数据 & 步骤 3: 计算向量
        if self.context["vector"] is None:
            self._step_fetch_and_compute()

        # 步骤 4: 向量搜索
        self._step_search()

        return self.context["recommendations"]

    def _step_check_database(self):
        """Stage 1: 查库"""
        cached_vector = db.find_vector_by_bvid(self.bvid)
        if cached_vector:
            print("⚡ [Pipeline] 命中数据库缓存，跳过计算。")
            self.context["vector"] = cached_vector

    def _step_fetch_and_compute(self):
        """Stage 2 & 3: 爬虫 + 模型"""
        print("🐢 [Pipeline] 未命中缓存，启动实时计算流程...")
        # 2.1 爬取
        danmaku_list = CrawlerService.fetch_danmaku(self.bvid)
        # 2.2 计算
        vector = model_service.predict(danmaku_list)
        self.context["vector"] = vector

        # (可选) 2.3: 这里可以把新计算的结果存回 database.json，实现“越用越快”

    def _step_search(self):
        """Stage 4: 相似度匹配"""
        if self.context["vector"]:
            results = db.search_similar(
                self.context["vector"],
                top_k=5,
                exclude_bvid=self.bvid
            )
            self.context["recommendations"] = results