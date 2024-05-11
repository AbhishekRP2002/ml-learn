import requests

if __name__ == "__main__":
    input_text = "Ray Serve eases the pain of model serving"
    try:
        result = requests.get("http://127.0.0.1:8265/sentiment", data=input_text).text
    except Exception as e:
        print("Error during request: {}".format(e))
        
    print("Result for '{}': {}".format(input_text, result))