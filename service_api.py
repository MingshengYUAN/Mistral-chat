from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request
import time
from flasgger import Swagger

# the device to load the model onto
device = "cuda:6" 
app = Flask(__name__)
templete = {
  "swagger": "2.0",
  "info": {
    "title": "Mistral Chat Completion Model API",
    "description": "Mistral Chat Completion Model API",
    "version": "0.0.1"
  },
  "tags": [
    {
      "name": "Input",
      "description": "Input"
    }
  ],
  "paths": {
    "/chat_completion": {
      "post": {
        "tags": [
          "chat_completion"
        ],
        "summary": "Chat Output",
        "description": "",
        "consumes": [
          "application/json"
        ],
        "produces": [
          "application/json"
        ],
        "parameters": [
          {
            "in": "body",
            "name": "body",
            "description": "input",
            "required": True,
            "schema": {
              "$ref": "#/definitions/prompt"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Response",
            "schema": {
              "$ref": "#/definitions/response"
            }
          }
        }
      }
    }
  },
  "definitions": {
    "prompt": {
      "type": "object",
      "required": [
        "messages"
      ],
      "properties": {
        "messages": {
          "type": "string",
          "items": {
            "type": "string"
          },
          "example": [{"role": "user", "content": "What is your favourite condiment?"},{"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},{"role": "user", "content": "Do you have mayonnaise recipes?"}]
        }
      }
    },
    "response": {
      "type": "object",
      "properties": {
        "response": {
          "items": {
            "type": "string"
          },
          "example": " Yes, I happen to have a recipe for a classic French mayonnaise that I'd love to share with you! Here it is:\n\nIngredients:\n\n* 2 egg yolks\n* 1 tablespoon Dijon mustard\n* 1 tablespoon white wine vinegar\n* 2 tablespoons water\n* 1/2 cup vegetable oil\n* Salt and freshly ground black pepper, to taste\n\nDirections:\n\n1. In a blender or food processor, whir together the egg yolks, Dijon mustard, white wine vinegar, and water until smooth.\n2. With the blender running, gradually pour in the vegetable oil in a steady stream until the mixture thickens.\n3. Taste and adjust seasoning if necessary.\n4. Serve immediately or place in a bowl with a lid and refrigerate until ready to use.\n\nI hope you enjoy this recipe!</s>"
        },
        "status": {
          "items": {
            "type": "string"
          },
          "example": "success"
        },
        "running_time": {
          "items": {
            "type": "number"
          },
          "example": "0.325542"
        }
      }
    },
    "ApiResponse": {
      "type": "object",
      "properties": {
        "code": {
          "type": "integer",
          "format": "int32"
        },
        "type": {
          "type": "string"
        },
        "message": {
          "type": "string"
        }
      }
    }
  }
}
swagger = Swagger(app, template=templete)

model = AutoModelForCausalLM.from_pretrained("/data-gpu02/home/mingsheng.yuan/yma/LLM/LLAMA2/Mistral/7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("/data-gpu02/home/mingsheng.yuan/yma/LLM/LLAMA2/Mistral//7B-Instruct-v0.1")

model.to(device)

@app.route("/chat_completion", methods=['POST'])
def messages():
    start = time.time()
    data = request.get_json()
    print(data)
    encodeds = tokenizer.apply_chat_template(data['messages'], return_tensors="pt")
    # encodeds = tokenizer(data['prompt'], return_tensors="pt").input_ids
    model_inputs = encodeds.to(device)
    
    generated_ids = model.generate(model_inputs, max_new_tokens=512, top_p=0.9, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    response = decoded[0].split("[/INST]")[-1]
    ret = {"response": response.strip('</s>'), "status": 'success', "running_time": float(time.time() - start)}
    return ret

if __name__=="__main__":
  app.run(port=3093, host="0.0.0.0", debug=False)