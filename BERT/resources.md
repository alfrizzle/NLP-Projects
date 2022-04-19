# Bert/Transformers + SageMaker Resources

## Pushing Docker Images to ECR

* https://docs.aws.amazon.com/AmazonECR/latest/userguide/image-push.html#image-push-iam
* https://transang.me/aws-roles-to-push-docker-image-to-elastic-container-registry/

## Train Language Model from Scratch
* https://huggingface.co/blog/how-to-train
* https://github.com/aws/amazon-sagemaker-examples/tree/main/training/distributed_training/pytorch/model_parallel/bert
* https://github.com/HerringForks/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT
* https://github.com/aws/amazon-sagemaker-examples/tree/main/training/distributed_training/pytorch

## Fine-tuning a PyTorch BERT model and deploying it with Amazon Elastic Inference on Amazon SageMaker

* https://aws.amazon.com/blogs/machine-learning/fine-tuning-a-pytorch-bert-model-and-deploying-it-with-amazon-elastic-inference-on-amazon-sagemaker/

* https://github.com/aws-samples/amazon-sagemaker-bert-pytorch

## Fine-tune and host Hugging Face BERT models on Amazon SageMaker

* https://aws.amazon.com/blogs/machine-learning/fine-tune-and-host-hugging-face-bert-models-on-amazon-sagemaker/

* https://github.com/aws-samples/finetune-deploy-bert-with-amazon-sagemaker-for-hugging-face

## Bring your own data to classify news with Amazon SageMaker and Hugging Face

* https://aws.amazon.com/blogs/machine-learning/bring-your-own-data-to-classify-news-with-amazon-sagemaker-and-hugging-face/

* https://github.com/aws-samples/classify-news-amazon-sagemaker-hugging-face

## Sentiment Classification with Amazon SageMaker

* https://github.com/brent-lemieux/sagemaker_train_demo

* https://towardsdatascience.com/getting-started-with-sagemaker-for-model-training-512b75eae7d7

## Training and Deploying a Sentiment Analysis PyTorch and Hugging Face Models in Amazon SageMaker

* https://github.com/JayThibs/aws-sagemaker-deploy-sentiment-analysis/tree/65d9753a98320adf9a22c23c93be7e07dc1bd741

* [General Outline of Project -- Notebook](https://github.com/JayThibs/aws-sagemaker-deploy-sentiment-analysis/blob/65d9753a98320adf9a22c23c93be7e07dc1bd741/SageMaker%20Project.ipynb)

## What's difference between tokenizer.encode and tokenizer.encode_plus in Hugging Face

* https://stackoverflow.com/questions/61708486/whats-difference-between-tokenizer-encode-and-tokenizer-encode-plus-in-hugging

## Getting Started with SageMaker for Model Training

* https://towardsdatascience.com/getting-started-with-sagemaker-for-model-training-512b75eae7d7

* https://github.com/brent-lemieux/sagemaker_train_demo/blob/main/src/model.py

## How to Create and Deploy Custom Python Models to SageMaker

https://www.predictifsolutions.com/tech-blog/how-to-custom-models-sagemaker/

## Deploy a Hugging Face Transformers Model from S3 to Amazon SageMaker
* https://www.youtube.com/watch?v=pfBGgSGnYLs

### Create a Amazon SageMaker endpoint with a trained model.
https://github.com/aws/sagemaker-huggingface-inference-toolkit


```
Example:

from sagemaker.huggingface import HuggingFaceModel

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    transformers_version='4.6',
    pytorch_version='1.7',
    py_version='py36',
    model_data='s3://my-trained-model/artifacts/model.tar.gz',
    role=role,
)

# deploy model to SageMaker Inference
huggingface_model.deploy(initial_instance_count=1,instance_type="ml.m5.xlarge")
```

## Recommended model for sentiment analysis
* distilbert-base-uncased-finetuned-sst-2-english
* https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
