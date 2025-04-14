
import torch
from torch.utils.data import DataLoader
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

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

from .encoder import VideoEncoder
from .decoder import VideoDecoder
from .simulate import simulate_transmission

class XNNPackModel:
    '''
    Export models to XNNPACK format and test them

    Parameters
    ----------
    enc_model_path : str
        Path to the encoder model
    dec_model_path : str
        Path to the decoder model
    export_dir : str
        Directory to save the exported model
    plot_dir : str
        Directory to save the plots
    quantize : bool
        Whether to quantize the model or not

    Attributes
    ----------
    export_dir : str
        Directory to save the exported model
    plot_dir : str
        Directory to save the plots
    enc_export_path : str
        Path to the exported encoder model
    dec_export_path : str
        Path to the exported decoder model
    encode : Executorch method
        Executorch method for encoding
    decode : Executorch method
        Executorch method for decoding

    Methods
    -------
    quantize_xnnpack(model, inputs) -> Module
        Quantize the model using Executorch
    export_xnnpack(model_path, quantize) -> str
        Export the model to XNNPACK format
    test_export(loader) -> None
        Test the exported XNNPACK model
    '''
    def __init__(
        self,
        ch_network:int,
        ch_compress:int,
        enc_model_path:str, 
        dec_model_path:str, 
        export_dir:str, 
        plot_dir:str,
        quantize:bool=True
    ):
        self.export_dir = export_dir
        self.plot_dir = plot_dir

        enc_model = VideoEncoder(ch_network, ch_compress)
        enc_model.load(enc_model_path)

        dec_model = VideoDecoder(ch_network, ch_compress)
        dec_model.load(dec_model_path)

        self.enc_export_path = self.export_xnnpack(enc_model, quantize)
        self.dec_export_path = self.export_xnnpack(dec_model, quantize)

        self.encode = None
        self.decode = None

    @staticmethod
    def quantize_xnnpack(model, inputs):
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

    def export_xnnpack(self, model:torch.nn.Module, quantize:bool=True):
        '''
        Transform, lower, and (optionally) quantize the model for XNNPACK backend

        Parameters
        ----------
        model : Module
            The model to be exported
        quantize : bool
            Whether to quantize the model or not

        Returns
        -------
        str
            Path to the exported model

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
        # Load the model
        model.eval()
        model.to(torch.device("cpu"))

        # Create a sample input tensor
        inputs = (torch.randn(1, 480, 640, 3),)

        # Create a GraphModule from the model
        export = torch.export.export_for_training(model, inputs, strict=True)
        model = export.module()
        
        # Quantize the model
        if quantize:
            model = self.quantize_xnnpack(model, inputs)
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
        save_pte_program(exec_prog, name, self.export_dir)  # Save the Executorch program

        return f"{self.export_dir}/{name}.pte"

    def test_export(self, loader:DataLoader):
        '''
        Test the exported XNNPACK model

        Parameters
        ----------
        loader : DataLoader
            DataLoader for the test dataset
        '''
        # Get the Executorch runtime environment
        runtime = Runtime.get()

        # Load the exported encoder
        program = runtime.load_program(self.enc_export_path)
        encode = program.load_method("forward")

        # Load the exported decoder
        program = runtime.load_program(self.dec_export_path)
        decode = program.load_method("forward")

        # Simulate transmission of the data
        simulate_transmission(loader, encode, decode, self.plot_dir)
        