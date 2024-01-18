import json

def retrieve_token():
    """
    Retrieve the user's token from the file api-token.json. It has the form:
    {
        "token":"..."
    }
    """

    with open("./api-token.json", "r") as file:
        
        data = json.load(file)
        print(data["token"])
        return data["token"]
