from logging import getLogger
from pathlib import Path
from typing import Self, cast

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from numpy import array
from numpy import float32 as np_float
from numpy.typing import NDArray
from onnxruntime import InferenceSession
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pydantic import BaseModel, Field
from torch import sigmoid
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline

logger = getLogger("Hiring Branch Logger")


class Model:
    def __init__(self: Self) -> None:
        super().__init__()
        self.classifier: InferenceSession
        self.pipe: FeatureExtractionPipeline

    def prepare(self: Self) -> None:
        classifier_onnx_path = Path(__file__).parent.joinpath(
            "models",
            "classifier.onnx",
        )
        bert_onnx_dir_path = Path(__file__).parent.joinpath("models", "bert")
        if not classifier_onnx_path.exists():
            msg = (
                f"{classifier_onnx_path} does not exist."
                "Please run cli optimize model before serving."
            )
            raise FileNotFoundError(msg)

        self.classifier = InferenceSession(
            str(classifier_onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        bert = cast(
            PreTrainedModel,
            ORTModelForFeatureExtraction.from_pretrained(
                bert_onnx_dir_path,
            ),
        )

        tokenizer = cast(
            PreTrainedTokenizer,
            AutoTokenizer.from_pretrained(
                Path(__file__).parent.joinpath("models", "bert"),
            ),
        )
        self.pipe = FeatureExtractionPipeline(model=bert, tokenizer=tokenizer)

    def infer(self: Self, text: str) -> float:
        embeddings = cast(list[list[list[float]]], self.pipe(text))
        embeddings_sentence: list[list[float]] = (
            torch.tensor(embeddings).sum(dim=1).tolist()
        )
        logits: NDArray[np_float] = array(
            cast(
                list[list[NDArray[np_float]]],
                self.classifier.run(
                    ["logits"],
                    input_feed={"embedding": embeddings_sentence},
                ),
            ),
        )
        result = sigmoid(torch.tensor(logits).squeeze())
        return cast(float, result.item())


class CustomApp(FastAPI):
    def __init__(self: Self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.model: Model


app = CustomApp(
    title="hiring_branch",
    openapi_tags=[
        {
            "name": "hiring branch",
            "description": "hiring branch API for passive voice",
        },
    ],
)

app.model = Model()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Exception in {request.url.path}: {exc!s}")
    return JSONResponse({"error": exc}, status_code=500)


@app.get("/", tags=["docs"])
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


class InferencePayload(BaseModel):
    text: str = Field(
        "The bananas are eaten by me.",
        description="The text to classify.",
        max_length=512,
    )


@app.post("/inference")
async def inference(payload: InferencePayload) -> JSONResponse:
    results = app.model.infer(payload.text)
    return JSONResponse({"logits": results})


def run_app() -> None:
    app.model.prepare()
    uvicorn.run(app, port=5000, log_level="info", loop="uvloop")
