from fastapi.testclient import TestClient

from hiring_branch.server import InferencePayload, app


def test_infer():
    app.model.prepare()
    client = TestClient(app)
    response = client.post(
        "/inference", json=InferencePayload(text="The bananas are eaten by me.").dict()
    )

    assert response.status_code == 200
    assert list(response.json().keys()) == ["logits"]
    assert isinstance(response.json()["logits"], float)
