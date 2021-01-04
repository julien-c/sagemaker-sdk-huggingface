from getpass import getpass
import os
import sys
import time

import boto3
import sagemaker
from transformers.hf_api import HfApi, HfFolder



# Tweak sys.path to import custom estimator
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(SRC_DIR)

from huggingface.estimator import HuggingFace




role_name = "SageMakerRole"

iam = boto3.client("iam")
sess = sagemaker.Session()
# get role arn
role = iam.get_role(RoleName=role_name)["Role"]["Arn"]


DEFAULT_BUCKET_ID = sess.default_bucket()
# list objects in s3 under datsets/
print(sess.list_s3_files(DEFAULT_BUCKET_ID, "datasets/"))


training_input_path = f"s3://{DEFAULT_BUCKET_ID}/datasets/imdb/train"
test_input_path = f"s3://{DEFAULT_BUCKET_ID}/datasets/imdb/test"


if HfFolder.get_token() is None:
    # User is not currently logged in, log them in now.
    username = input("Username: ")
    password = getpass()
    token = HfApi().login(username, password)
    HfFolder.save_token(token)


huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./examples/scripts",
    sagemaker_session=sess,
    base_job_name="huggingface-sdk-extension",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    framework_version={"transformers": "4.1.1", "datasets": "1.1.3"},
    py_version="py3",
    hyperparameters={
        "epochs": 1,
        "train_batch_size": 32,
        "model_name": "distilbert-base-uncased",
        "huggingface_token": HfFolder.get_token(),
        "hub_repo_name": f"distilbert-sagemaker-{int(time.time())}",
    },
)

huggingface_estimator.fit({
    'train': training_input_path,
    'test': test_input_path,
})


print(
    huggingface_estimator.latest_training_job.name
)


print()
