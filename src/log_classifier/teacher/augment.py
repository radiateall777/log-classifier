import random
import re
from typing import List


class TextCodeAugmenter:
    """
    Label-preserving text/code augmentation for Stage 2 training.
    Provides various rule-based augmentations to improve model robustness.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(self.seed)

        # Pre-compile some regex patterns
        # 1. Role markers
        self.role_pattern = re.compile(
            r"\b(user|assistant|Objective|Task|Details|Example)\s*:\s*", re.IGNORECASE
        )
        
        # 2. Markdown noise (```java, ```python, etc.)
        self.md_pattern = re.compile(r"```[a-zA-Z]*\n|```", re.IGNORECASE)

        # 3. Code variables (very basic heuristic: word matching)
        # Avoid keywords
        self.keywords = {
            "public", "private", "protected", "class", "static", "void", "int", "String",
            "return", "new", "if", "else", "for", "while", "import", "package", "def",
            "self", "import", "from", "as", "True", "False", "None", "try", "except",
            "finally", "const", "let", "var", "function", "=>", "await", "async"
        }
        
        # 4. Code comments
        self.line_comment_pattern = re.compile(r"//.*$", re.MULTILINE)
        self.block_comment_pattern = re.compile(r"/\*.*?\*/", re.DOTALL)
        self.python_comment_pattern = re.compile(r"#.*$", re.MULTILINE)

        # 8. Term swaps
        self.term_swaps = {
            "function": "方法",
            "方法": "function",
            "class": "类",
            "类": "class",
            "variable": "变量",
            "变量": "variable",
            "string": "字符串",
            "字符串": "string",
            "error": "错误",
            "错误": "error",
        }
        
        self.harmless_noises = [
            "Example usage:",
            "The following is a simple implementation.",
            "Here is the code:",
            "Note:",
            "Solution:"
        ]

        self.augmentations = [
            self.remove_role_markers,
            self.remove_markdown_noise,
            self.rename_code_variables,
            self.remove_code_comments,
            self.normalize_code_whitespace,
            self.truncate_assistant_explanation,
            self.add_harmless_noise,
            self.cn_en_term_swap,
        ]

    def _safe_augment(self, func, text: str) -> str:
        try:
            res = func(text)
            if not res.strip():
                return text
            return res
        except Exception:
            return text

    def augment(self, text: str, n: int = 1) -> List[str]:
        """
        Apply a random augmentation. If it results in empty string, return original.
        Returns `n` augmented versions.
        """
        results = []
        for _ in range(n):
            aug_func = self.rng.choice(self.augmentations)
            aug_text = self._safe_augment(aug_func, text)
            results.append(aug_text)
        return results

    def remove_role_markers(self, text: str) -> str:
        """Remove or weaken common role markers."""
        return self.role_pattern.sub(" ", text).strip()

    def remove_markdown_noise(self, text: str) -> str:
        """Remove markdown fences while keeping code content."""
        return self.md_pattern.sub("", text).strip()

    def rename_code_variables(self, text: str) -> str:
        """Light regex-based renaming of local variable-like identifiers."""
        words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        candidates = [w for w in words if w not in self.keywords and len(w) > 3]
        if not candidates:
            return text
        
        target = self.rng.choice(candidates)
        new_name = target + "_var"
        
        # Replace whole words only
        pattern = re.compile(rf"\b{re.escape(target)}\b")
        return pattern.sub(new_name, text)

    def remove_code_comments(self, text: str) -> str:
        """Remove // and /* */ and # comments."""
        text = self.block_comment_pattern.sub("", text)
        text = self.line_comment_pattern.sub("", text)
        text = self.python_comment_pattern.sub("", text)
        return text.strip()

    def normalize_code_whitespace(self, text: str) -> str:
        """Change indentation/multiple spaces without deleting content."""
        # Replace multiple spaces with a single space or tab
        if self.rng.random() > 0.5:
            return re.sub(r" +", " ", text)
        else:
            return re.sub(r" {2,}", "\t", text)

    def truncate_assistant_explanation(self, text: str) -> str:
        """If text contains 'assistant:', drop part of explanatory prose after code block."""
        parts = text.split("assistant:", 1)
        if len(parts) == 2:
            assistant_text = parts[1]
            code_blocks = re.split(r"(```.*?```)", assistant_text, flags=re.DOTALL)
            if len(code_blocks) >= 3:
                # E.g. [prose, code, prose]
                # Drop the trailing prose
                new_assistant = "".join(code_blocks[:-1])
                return parts[0] + "assistant:" + new_assistant
        return text

    def add_harmless_noise(self, text: str) -> str:
        """Add small harmless phrases."""
        noise = self.rng.choice(self.harmless_noises)
        if self.rng.random() > 0.5:
            return noise + "\n" + text
        else:
            return text + "\n" + noise

    def cn_en_term_swap(self, text: str) -> str:
        """Small safe replacements between EN/CN terms."""
        for term, replacement in self.term_swaps.items():
            if term in text:
                # Replace just one occurrence to avoid over-perturbation
                text = text.replace(term, replacement, 1)
                break
        return text
