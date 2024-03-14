from pathlib import Path
from tempfile import mkdtemp

from boto3 import resource
from wandb.apis import PublicApi as wandb_api


def download_weights(s3_or_wb_path: str) -> Path:
    """
    Download the weights from either a s3 bucket or a weight and biases artifact name.

    Parameters
    ----------
    s3_or_wb_path : str
        The path to the weights and biases artifact name or the s3 URI.

    Returns
    -------
    Path
        A Path to the PytorchLightning Model stored in a temporary directory.

    Raises
    ------
    AssertionError
        If the weights and biases artifact name or the s3 URI is not a .ckpt model path.
    """
    temp_dir = Path(mkdtemp())

    if "s3://" not in s3_or_wb_path:
        model_artifact = wandb_api().artifact(
            f"{s3_or_wb_path}:best",
            type="model",
        )
        checkpoint_folder = model_artifact.download(root=str(temp_dir))
        return Path(checkpoint_folder, "model.ckpt")

    s3_or_wb_path = s3_or_wb_path.replace("s3://", "")
    bucket_name = s3_or_wb_path.split("/")[0]

    if not s3_or_wb_path.split("/")[-1].endswith(".ckpt"):
        msg = (
            "The S3 URI needs to be a ckpt pytorch lightning checkpoint...\n"
            f"Got {s3_or_wb_path}",
        )
        raise AssertionError(msg)

    s3_bucket = resource("s3").Bucket(bucket_name)
    s3_bucket.download_file(
        s3_or_wb_path.replace(f"{bucket_name}/", ""),
        str(Path(temp_dir, "model.ckpt")),
    )
    return Path(temp_dir, "model.ckpt")
