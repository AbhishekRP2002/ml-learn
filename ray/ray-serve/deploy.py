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