
import sys
import copy

import torch
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

# Import debugging tools
from executorch.devtools import generate_etrecord

# Import the custom modules
# from .model import VideoModel
from .encoder import VideoEncoder
from .decoder import VideoDecoder

class ExecutorchModel:
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
    load_exported_models() -> tuple
        Load the exported encoder and decoder models
    '''
    def __init__(
        self,
        step:int,
        encoder:VideoEncoder,
        decoder:VideoDecoder,
        c_network:int,
        c_compress:int,
        export_dir:str,
        control_dir:str,
        remote_dir:str,
        quantize:bool=True,
        device:str="cpu",
    ):
        self.step = step
        self.export_dir = export_dir
        verification_error = True
        
        # Move the models to the CPU
        encoder.to("cpu").eval()
        decoder.to("cpu").eval()

        # Get an Executorch runtime
        runtime = Runtime.get()

        # Prepare the encoder
        x = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)
        self.enc_export_path = self.export_xnnpack(encoder, (x,), "img_enc", remote_dir, quantize)
        enc_program = runtime.load_program(self.enc_export_path)
        if verification_error: sys.stdout.write("\033[F\033[K")
        self.enc_method = enc_program.load_method("forward")

        # Prepare the decoder
        z_string = torch.randint(-128, 127, (c_network * 8 * 10,), dtype=torch.int8)
        y_string = torch.randint(-128, 127, (c_compress * 30 * 40,), dtype=torch.int8)
        self.dec_export_path = self.export_xnnpack(decoder, (z_string, y_string), "img_dec", control_dir, quantize)
        dec_program = runtime.load_program(self.dec_export_path)
        if verification_error: sys.stdout.write("\033[F\033[K")
        self.dec_method = dec_program.load_method("forward")

        # Move the models back to original state
        encoder.to(device).train()
        decoder.to(device).train()

    def encoder(self, x:torch.ByteTensor) -> tuple[torch.CharTensor, torch.CharTensor]:
        return self.enc_method.execute([x])

    def decoder(self, z_string:torch.CharTensor, y_string:torch.CharTensor) -> torch.ByteTensor:
        return self.dec_method.execute((z_string, y_string))[0]

    def export_xnnpack(self, model:torch.nn.Module, inputs:tuple[torch.Tensor], name:str, app_dir:str=None, quantize:bool=True) -> str:
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

        # Get export program for the model
        ep = torch.export.export_for_training(model, inputs, strict=True)

        # Quantize the model
        if quantize:
            quantizer = XNNPACKQuantizer()
            qparams = get_symmetric_quantization_config()
            quantizer.set_global(qparams)

            m = prepare_pt2e(ep.module(), quantizer)
            m(*inputs)
            m = convert_pt2e(m)

            ep = torch.export.export_for_training(m, inputs, strict=True)


        # Transform the model to edge dialect, then lower to the XNNPACK backend
        edge = to_edge_transform_and_lower(
            ep,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=(not quantize),
                _skip_dim_order=True
            ),
        )

        edge_copy = copy.deepcopy(edge)

        et_program = edge.to_executorch(config=ExecutorchBackendConfig(extract_delegate_segments=False))

        # Set the file name for the exported model
        name = f"{name}_xnnpack_{"q8" if quantize else "fp32"}"

        # Generate an ET record for the model
        generate_etrecord(f"{self.export_dir}/{name}_{self.step}.bin", edge_copy, et_program)

        # Save the executable in the export directory
        if self.export_dir is not None:
            save_pte_program(et_program, f"{name}_{self.step}", self.export_dir)

        # Put the executable in the application directory
        if app_dir is not None:
            save_pte_program(et_program, name, app_dir)

        return f"{self.export_dir}/{name}_{self.step}.pte"
