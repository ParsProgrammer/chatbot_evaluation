from pydantic import BaseModel, Field
from typing import List


class TestCase(BaseModel):
    test_id: str = Field(..., min_length=1)
    conversation: List[str] = Field(..., min_length=1)
    expected_intents: List[str] = Field(..., min_length=1)
    expected_response_keywords: List[List[str]] = Field(default_factory=list)

    def normalized(self) -> "TestCase":
        kw = list(self.expected_response_keywords)
        if len(kw) < len(self.conversation):
            kw.extend([[] for _ in range(len(self.conversation) - len(kw))])
        return self.model_copy(update={"expected_response_keywords": kw})


class Dataset(BaseModel):
    tests: List[TestCase] = Field(default_factory=list)

    @classmethod
    def from_json_obj(cls, obj: object) -> "Dataset":
        # Supports either [{"test_id":...}, ...] or {"tests":[...]}
        if isinstance(obj, list):
            return cls(tests=[TestCase.model_validate(x) for x in obj])
        if isinstance(obj, dict) and "tests" in obj:
            return cls.model_validate(obj)
        raise ValueError("Dataset JSON must be a list of test cases or an object with 'tests'.")