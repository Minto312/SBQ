echo $1
if [ $1 == "encode" ]; then

    protoc --encode onnx.ModelProto onnx.proto < model.txt > model.onnx

elif [ $1 == "decode" ]; then

    protoc --decode onnx.ModelProto onnx.proto < model.onnx > model.txt

else
    echo "Usage: onnx_coder.sh [encode|decode] [input_file] [output_file] [model_name] [model_version] [model_type] [model_framework] [model_domain] [model_doc_string]"
    echo "Example: onnx_coder.sh encode model.onnx model.jnk model 1 onnx onnx ai.onnx.ml"
fi