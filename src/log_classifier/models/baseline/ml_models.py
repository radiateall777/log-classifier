"""传统机器学习 baseline 的模型构建（包含特征提取与分类器）。"""

from typing import Tuple, Any, Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import xgboost as xgb


def build_tfidf_vectorizer(
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """构建统一的 TF-IDF 向量化器"""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        dtype=np.float32,
    )

class EmbeddingVectorizer:
    """包装 sentence-transformers 为类似 sklearn 的向量化器风格"""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None  # 懒加载以避免被 pickle

    def _load(self):
        if self._model is None:
            # 在类内部导入，避免强依赖和重复加载
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def fit_transform(self, X: List[str], y=None):
        self._load()
        return self._model.encode(X, convert_to_numpy=True, show_progress_bar=True)

    def transform(self, X: List[str]):
        self._load()
        return self._model.encode(X, convert_to_numpy=True, show_progress_bar=False)

    def __getstate__(self):
        # 序列化时由于模型太大且无法跨环境被存活，我们只保存名字
        return {"model_name": self.model_name}

    def __setstate__(self, state):
        self.model_name = state["model_name"]
        self._model = None


def build_ml_classifier(method: str, seed: int, **kwargs: Any) -> Any:
    """根据方法名和参数构建分类器实例"""
    if method in ("tfidf_lr", "embed_lr"):
        return LogisticRegression(
            max_iter=1000,
            C=kwargs.get("C", 1.0),
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
    if method in ("tfidf_svm", "embed_svm"):
        return LinearSVC(
            C=kwargs.get("C", 1.0),
            max_iter=5000,
            class_weight="balanced",
            random_state=seed,
            dual="auto",
        )
    if method == "tfidf_nb":
        return MultinomialNB(
            alpha=kwargs.get("alpha", 1.0)
        )
    if method in ("tfidf_xgb", "embed_xgb"):
        return xgb.XGBClassifier(
            learning_rate=kwargs.get("learning_rate", 0.1),
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 6),
            objective="multi:softmax",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    
    raise ValueError(f"不支持的 ML 方法: {method}")

# 超参搜索网格定义
ML_PARAM_GRIDS: Dict[str, Dict[str, list]] = {
    "tfidf_lr": {"C": [0.01, 0.1, 1.0, 10.0]},
    "tfidf_svm": {"C": [0.01, 0.1, 1.0, 10.0]},
    "tfidf_nb": {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
    "tfidf_xgb": {
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 300],
    },
    "embed_lr": {"C": [0.1, 1.0, 10.0]},
    "embed_svm": {"C": [0.1, 1.0, 10.0]},
    "embed_xgb": {
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 300],
    },
}
