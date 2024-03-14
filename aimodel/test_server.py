from typing import cast

from fastapi.testclient import TestClient
from requests import Response

from aimodel.server import InferencePayload, app


def test_infer() -> None:
    app.model.prepare()
    client = TestClient(app)
    response = cast(
        Response,
        client.post(
            "/inference",
            json=InferencePayload(text="The bananas are eaten by me.").model_dump(),
        ),
    )

    ok_status_code = 200
    assert response.status_code == ok_status_code
    assert list(response.json().keys()) == ["logits"]
    assert isinstance(response.json()["logits"], float)
