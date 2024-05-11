import ray
import requests
from starlette.requests import Request
from ray import serve
from transformers import pipeline
# from model import SentimentDeployment

@serve.deployment(name="sentiment-classification")
class SentimentDeployment:
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")

    async def __call__(self, request : Request):
        await request.body()
        result= self.classifier(request.query_params["text"])[0]
        return result

# Connect to the running Ray Serve instance.
if __name__ == "__main__":
    ray.init(address='auto', namespace="serve-example", ignore_reinit_error=True)
    serve.start(detached=True)

    # Deploy the model.
    app = SentimentDeployment.bind()
    serve.run(app, route_prefix="/sentiment")
    print("Model deployed as an API.")
    # test the api request
    input_text = "Ray Serve reduces the complexity of model serving" # false negative
    try:
        result = requests.get("http://localhost:8000/sentiment", params = {"text" : input_text}).json()
    except Exception as e:
        print("Error during request: {}".format(e))
        
    print("Result for '{}': {}".format(input_text, result))
    ray.shutdown()




# import requests
# from starlette.requests import Request
# from typing import Dict

# from transformers import pipeline

# from ray import serve


# # 1: Wrap the pretrained sentiment analysis model in a Serve deployment.
# @serve.deployment
# class SentimentAnalysisDeployment:
#     def __init__(self):
#         self._model = pipeline("sentiment-analysis")

#     def __call__(self, request: Request) -> Dict:
#         return self._model(request.query_params["text"])[0]


# # 2: Deploy the deployment.
# serve.run(SentimentAnalysisDeployment.bind(), route_prefix="/")

# # 3: Query the deployment and print the result.
# print(
#     requests.get(
#         "http://localhost:8000/", params={"text": "Ray Serve is great!"}
#     ).json()
# )
