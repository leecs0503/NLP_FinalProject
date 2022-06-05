import base64
import json

with open("COCO_val2014_000000088902.jpg", "rb") as img_file:
    b64_string = base64.b64encode(img_file.read())
question = "what direction is the traffic light indicating?"

print(json.dumps({
    "base64image": b64_string.decode(),
    "question": question,
},indent=4))
