
import argparse
import os
import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower
)
from executorch.extension.export_util.utils import save_pte_program
from executorch.runtime import Runtime
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

def quantize(model, inputs):
    '''
    Quantize the model using Executorch

    Parameters
    ----------
    model : torch.nn.Module
        The model to be quantized
    inputs : torch.Tensor
        Sample input tensor for the model

    Returns
    -------
    quantized_model : torch.nn.Module
        The quantized model
    '''
    quantizer = XNNPACKQuantizer()
    operator_config = get_symmetric_quantization_config(
        is_per_channel=True,
        is_dynamic=False,
    )
    quantizer.set_global(operator_config)
    m = prepare_pt2e(model, quantizer)
    m(*inputs)
    m = convert_pt2e(m)
    return m

def export_xnnpack(model_path:str, model_dir:str, quantize:bool=True):
    '''
    Export the model to XNNPACK format

    Parameters
    ----------
    model_path : str
        Path to the model file
    model_dir : str
        Directory to save the exported model
    quantize : bool
        Whether to quantize the model or not

    Notes
    -----
    Example usage of the exported XNNPACK model in C++:
    
    ```
    #include <executorch/extension/module/module.h>
    #include <executorch/extension/tensor/tensor.h>

    using namespace ::executorch::extension;

    // Load the model.
    Module module("/path/to/model.pte");

    // Create an input tensor.
    float input[1 * 3 * 480 * 640];
    auto tensor = from_blob(input, {1, 3, 480, 640});

    // Perform an inference.
    const auto result = module.forward(tensor);

    // Retrieve the output data.
    if (result.ok()) {
        const auto output = result->at(0).toTensor().const_data_ptr<float>();
    }
    ```
    '''
    model.eval()

    inputs = torch.randn(model.batch_size, 3, 480, 640)
    export = torch.export.export_for_training(model, inputs, strict=True)
    model = export.module()
    
    # Quantize the model
    if quantize:
        model = quantize(model, inputs)
        export = torch.export.export_for_training(model, inputs, strict=True)

    # Lower the model, then transform the model to Executorch backend
    edge = to_edge_transform_and_lower(           
        export, 
        partitioner=[XnnpackPartitioner()], # Create a partitioner for XNNPACK
        compile_config=EdgeCompileConfig(   # Set the compile configuration
            _check_ir_validity=False, 
            _skip_dim_order=True
        )
    )

    exec_prog = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

    name = f"{model.__class__.__name__}_XNNPack_{"q8" if quantize else "fp32"}"
    save_pte_program(exec_prog, name, model_dir)  # Save the Executorch program


def test_xnnpack(path: str, method: str, input: torch.Tensor):
    '''
    Test the exported XNNPACK model

    Parameters
    ----------
    path : str
        Path to the exported XNNPACK model
    method : str
        Method to be executed
    input : torch.Tensor
        Input tensor to the model | *(1, 3, 480, 640)*

    Returns
    -------
    outputs : list
        Output tensors from the model
    '''
    assert input.shape == (1, 3, 480, 640), f"Input shape must be (1, 3, 480, 640), got {input.shape}"
    runtime = Runtime.get()                 # Get the runtime instance
    program = runtime.load_program(path)    # Load the exported XNNPACK model
    method = program.load_method(method)    # Load the specified method
    outputs = method.execute([input])       # Execute the model with the input tensor
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and test XNNPACK model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the exported model")
    parser.add_argument("--quantize", action="store_true", help="Whether to quantize the model or not")
    args = parser.parse_args()

    # Ensure the model directory exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Export the model
    export_xnnpack(args.model_path, args.model_dir, args.quantize)