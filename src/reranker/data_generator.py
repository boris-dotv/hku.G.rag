"""
Reranker Training Data Generation Pipeline

Uses GLM-4.7 API to generate training data with:
- Query: Natural language question
- Chunk: Document text segment
- Keywords: Key terms extracted from chunk
- Score: Relevance score (1-10) with detailed rationale
"""

import json
import time
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class TrainingSample:
    """Single training sample for reranker"""
    query: str
    chunk: str
    keywords: List[str]
    score: float
    rationale: str

    def to_dict(self) -> dict:
        return asdict(self)


class GLM4Client:
    """GLM-4.7 API Client with retry logic"""

    def __init__(self, api_key: str, max_retries: int = 3):
        self.api_key = api_key
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.max_retries = max_retries

    def call(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Call GLM-4.7 API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                payload = {
                    "model": "glm-4.7",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except (requests.RequestException, requests.Timeout) as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"    API timeout/error, retrying in {wait_time}s... (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                else:
                    raise

    def call_with_json(self, messages: List[Dict], **kwargs) -> dict:
        """Call API and parse JSON response"""
        content = self.call(messages, **kwargs)

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Fallback: try to extract JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
            raise ValueError(f"Failed to parse JSON from response: {content}")


# ============================================
# Scoring Criteria (1-10 scale)
# ============================================
SCORING_CRITERIA = """
You are an expert information retrieval quality evaluator. Score the relevance between Query and Document on a scale of 1-10:

**Score 10 - PERFECT MATCH**: The document completely answers the query, contains all key concepts, is logically clear, and requires no additional information.

**Score 9 - EXCELLENT MATCH**: The document addresses the core question, contains most keywords, with minimal redundancy.

**Score 8 - GOOD MATCH**: The document is highly relevant to the query, contains core keywords, but may need slight supplementation.

**Score 7 - ABOVE AVERAGE**: The document partially answers the query, contains some keywords, but not comprehensive enough.

**Score 6 - AVERAGE MATCH**: The document has some relevance to the query, contains a few keywords, but requires significant supplementation.

**Score 5 - WEAK RELEVANCE**: The document mentions topics related to the query but doesn't directly answer the question.

**Score 4 - LOW RELEVANCE**: The document is thematically related but content doesn't match the query.

**Score 3 - VERY LOW RELEVANCE**: Only minimal content in the document relates to the query.

**Score 2 - MOSTLY IRRELEVANT**: The document has almost no relation to the query.

**Score 1 - COMPLETELY IRRELEVANT**: The document is completely unrelated or contains incorrect information.
"""


class RerankerDataGenerator:
    """
    Generate training data for Reranker fine-tuning using GLM-4.7

    Pipeline:
    1. Extract keywords from chunks
    2. Generate queries based on chunks (positive samples)
    3. Score relevance between queries and chunks
    4. Generate negative samples by mismatching queries and chunks
    """

    def __init__(self, api_key: str):
        self.client = GLM4Client(api_key)

    def extract_keywords(self, chunk: str, num_keywords: int = 8) -> List[str]:
        """Extract key technical terms/concepts from chunk"""
        prompt = f"""
Analyze the following academic text and extract {num_keywords} key technical terms or concepts.

Focus on:
- Technical terminology specific to the domain
- Core concepts that define the topic
- Important acronyms or named entities

Text:
{chunk[:1500]}

Return a JSON object with a "keywords" array:
{{"keywords": ["term1", "term2", ...]}}
"""
        messages = [{"role": "user", "content": prompt}]
        result = self.client.call_with_json(messages, temperature=0.3)
        return result.get("keywords", [])[:num_keywords]

    def generate_query(self, chunk: str, keywords: List[str]) -> str:
        """Generate a natural language query based on chunk content"""
        prompt = f"""
You are a student studying this course material. Generate a natural question that a student might ask based on the following content.

The question should:
- Be clear and specific
- Require understanding of the core concepts
- Be answerable from the given content
- Sound natural (like a real student question)

Content:
{chunk[:1500]}

Key concepts: {', '.join(keywords[:5])}

Return a JSON object with a "question" field:
{{"question": "your question here"}}
"""
        messages = [{"role": "user", "content": prompt}]
        result = self.client.call_with_json(messages, temperature=0.8)
        return result.get("question", "")

    def score_relevance(self, query: str, chunk: str, keywords: List[str]) -> tuple[float, str]:
        """Score the relevance between query and chunk (1-10)"""
        prompt = f"""{SCORING_CRITERIA}

**Task**: Evaluate the relevance between the following Query and Document

Query: {query}

Document: {chunk[:2000]}

Key Concepts in Document: {', '.join(keywords[:8])}

Return a JSON object with "score" (number 1-10) and "rationale" (brief explanation):
{{"score": 8, "rationale": "The document directly addresses the core question..."}}
"""
        messages = [{"role": "user", "content": prompt}]
        result = self.client.call_with_json(messages, temperature=0.3)
        return float(result.get("score", 5)), result.get("rationale", "")

    def generate_positive_sample(self, chunk: str) -> TrainingSample:
        """Generate a positive training sample from a chunk"""
        # 1. Extract keywords
        print("  [1/3] Extracting keywords...")
        keywords = self.extract_keywords(chunk)

        # 2. Generate query
        print("  [2/3] Generating query...")
        query = self.generate_query(chunk, keywords)

        # 3. Score relevance
        print("  [3/3] Scoring relevance...")
        score, rationale = self.score_relevance(query, chunk, keywords)

        return TrainingSample(
            query=query,
            chunk=chunk,
            keywords=keywords,
            score=score,
            rationale=rationale
        )

    def generate_negative_sample(self, query: str, chunk: str, keywords: List[str]) -> TrainingSample:
        """Generate a negative training sample (mismatched query-chunk)"""
        print("  [1/1] Scoring negative sample...")
        score, rationale = self.score_relevance(query, chunk, keywords)

        return TrainingSample(
            query=query,
            chunk=chunk,
            keywords=keywords,
            score=score,
            rationale=rationale
        )

    def generate_dataset(
        self,
        chunks: List[str],
        num_samples: int = 900,
        negative_ratio: float = 0.5
    ) -> List[TrainingSample]:
        """
        Generate full training dataset

        Args:
            chunks: List of document chunks
            num_samples: Target number of samples
            negative_ratio: Ratio of negative samples to positive samples
        """
        dataset = []
        positive_queries = []  # Store (query, keywords) from positive samples

        for i, chunk in enumerate(chunks):
            if len(dataset) >= num_samples:
                break

            print(f"\nGenerating positive sample {len(dataset) + 1}/{num_samples}")
            print(f"Chunk {i+1}/{len(chunks)}")

            try:
                # Generate positive sample
                positive = self.generate_positive_sample(chunk)
                dataset.append(positive)
                positive_queries.append((positive.query, positive.keywords))

                # Generate negative sample using a previous query
                if len(positive_queries) > 2 and len(dataset) < num_samples:
                    prev_query, prev_keywords = positive_queries[max(0, len(positive_queries) - 3)]
                    print(f"\nGenerating negative sample...")
                    negative = self.generate_negative_sample(prev_query, chunk, prev_keywords)
                    dataset.append(negative)

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"  Error generating sample: {e}")
                continue

        return dataset

    def save_dataset(self, dataset: List[TrainingSample], output_path: str):
        """Save dataset to JSONL file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in dataset:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        print(f"\nSaved {len(dataset)} samples to {output_path}")


# ============================================
# Demo / Test
# ============================================
if __name__ == "__main__":
    API_KEY = "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq"

    # Sample chunks for testing (will be replaced with actual parsed PDF chunks)
    SAMPLE_CHUNKS = [
        """Machine Learning vs Deep Learning

Machine Learning (ML) is a subset of artificial intelligence that uses statistical techniques to give computer systems the ability to "learn" from data, without being explicitly programmed.

Deep Learning (DL) is a specialized subset of machine learning that uses neural networks with many layers (deep neural networks) to model complex patterns in data. Key differences include:

1. Feature Engineering: ML requires manual feature selection, while DL learns features automatically
2. Data Requirements: DL typically needs much more data to perform well
3. Computational Power: DL requires more computational resources (GPUs)
4. Interpretability: ML models are generally more interpretable than DL models

Common ML algorithms include: Linear Regression, Decision Trees, Random Forest, SVM.
Common DL architectures include: CNNs, RNNs, Transformers.""",

        """Supervised Learning Algorithms

Supervised learning is a type of machine learning where the model learns from labeled training data. The algorithm learns a mapping function from input variables (X) to output variables (Y).

Main categories:
1. Classification: Predicting discrete labels (e.g., spam detection, image classification)
2. Regression: Predicting continuous values (e.g., house prices, temperature)

Popular Algorithms:

Decision Trees: Tree-like model where decisions are made based on feature values. Easy to interpret but prone to overfitting.

Random Forest: Ensemble method that combines multiple decision trees. Reduces overfitting and improves accuracy.

Support Vector Machines (SVM): Finds the optimal hyperplane that separates different classes. Effective in high-dimensional spaces.

Logistic Regression: Despite its name, it's used for classification. Predicts probability using logistic function.""",

        """Overfitting and Underfitting

Overfitting occurs when a model learns the training data too well, including noise and random fluctuations. This results in poor generalization to new, unseen data.

Signs of overfitting:
- High training accuracy, low test accuracy
- Model is overly complex relative to data size
- Large gap between training and validation performance

Solutions:
- More training data
- Simplify the model (fewer layers/parameters)
- Regularization (L1/L2 penalties)
- Dropout (for neural networks)
- Early stopping
- Cross-validation

Underfitting occurs when the model is too simple to capture the underlying patterns in the data.""",

        """Regression Analysis

Regression Analysis is a predictive modeling technique that estimates the relationship between a dependent (target) and an independent variable (predictor).

Three major uses for regression analysis are:
1. Determining the strength of predictors
2. Forecasting an effect
3. Trend forecasting

Types of Regression:
- Linear Regression: Models linear relationship between variables
- Polynomial Regression: Models non-linear relationships
- Logistic Regression: Used for binary classification
- Ridge/Lasso Regression: Includes regularization to prevent overfitting""",

        """Training and Test Data

In machine learning, it's essential to split data into training and test sets:

Training Set (typically 70-80%): Used to train the model and learn parameters.

Test Set (typically 20-30%): Used to evaluate how well the model generalizes to unseen data.

Validation Set: Sometimes used to tune hyperparameters during training.

Important: Never use test data for training! This leads to data leakage and overly optimistic performance estimates.

Cross-validation is a technique where data is split into k folds, and each fold serves as the test set once. This provides more reliable performance estimates."""
    ]

    generator = RerankerDataGenerator(API_KEY)

    # Generate a few samples for testing
    print("=" * 50)
    print("GENERATING TEST SAMPLES")
    print("=" * 50)

    dataset = generator.generate_dataset(SAMPLE_CHUNKS, num_samples=6)
    generator.save_dataset(dataset, "data/reranker_data/test_samples.jsonl")

    # Print summary
    print("\n" + "=" * 50)
    print("SAMPLE SUMMARY")
    print("=" * 50)
    for i, sample in enumerate(dataset, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Query: {sample.query}")
        print(f"Keywords: {', '.join(sample.keywords)}")
        print(f"Score: {sample.score}/10")
        print(f"Rationale: {sample.rationale}")
