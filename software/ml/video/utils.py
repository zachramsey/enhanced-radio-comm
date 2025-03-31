
import sys
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

def print_inline_every(iter, freq, term, msg):
    if iter % freq == 0 or iter == term - 1:
        if iter > 0: sys.stdout.write("\033[F\033[K")
        print(msg)

def tensor_to_image(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = image.permute(1, 2, 0) # C x H x W  -> H x W x C
    image = image.numpy()
    image = (image * 255).astype('uint8') # Assuming images are normalized to [0, 1]
    return image

def export_xnnpack(self, path):
    '''
    Export the model to XNNPACK format

    Parameters
    ----------
    path : str
        Path to save the model

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
    self.eval()                                         # Set the model to evaluation mode
    inputs = torch.randn(self.batch_size, 3, 480, 640)  # Create a sample input tensor
    export = torch.export.export(self, inputs)          # Export the model
    partitioner = [XnnpackPartitioner()]                # Create a partitioner for XNNPACK

    # Lower the model, then transform the model to Executorch backend
    et_program = to_edge_transform_and_lower(           
        export, 
        partitioner=partitioner
    ).to_executorch()                                   

    # Save the model to the specified path
    with open(path, 'wb') as f:
        f.write(et_program.buffer)


from executorch.runtime import Runtime

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