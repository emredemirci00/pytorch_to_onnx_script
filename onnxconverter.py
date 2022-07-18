
import sys
from transformers import AutoTokenizer
from txtai.pipeline import HFOnnx
import onnx

def main():
  print("first argument should be path to pytorch model ")
  print("second argument should be path where your onnx model is gonna be saved")
  tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
  tokenizer.save_pretrained(sys.argv[1])
  hfonnx = HFOnnx()
  model = hfonnx(sys.argv[1], "text-classification")
  onnx.save(model, sys.argv[2])
  print("onnx succesfully saved to path -->",sys.argv[2])

if __name__ == '__main__':
    main()