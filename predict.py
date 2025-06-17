import os
import sys
import tempfile
import shutil
import subprocess
from typing import List
import torch
import numpy as np
import cv2
from PIL import Image
from cog import BasePredictor, File, Path, Input

# Add OneFormer to path
sys.path.append("/OneFormer")

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

# Import from OneFormer demo
sys.path.append("/OneFormer/demo")
from defaults import DefaultPredictor
from visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        
        # Compile MultiScaleDeformableAttentionCUDA op
        ops_dir = "/OneFormer/oneformer/modeling/pixel_decoder/ops"
        print("Compiling MultiScaleDeformableAttention CUDA op...")

        # Copy the working CUDA file to the correct location
        cuda_file = "/OneFormer/oneformer/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu"
        working_cuda_file = "/src/ms_deform_attn_cuda.cu"
        
        print(f"Checking for working CUDA file at: {working_cuda_file}")
        if os.path.exists(working_cuda_file):
            print(f"Found working CUDA file, copying to: {cuda_file}")
            # Remove existing file if it exists
            if os.path.exists(cuda_file):
                os.remove(cuda_file)
            # Copy the working file
            shutil.copy(working_cuda_file, cuda_file)
            print(f"File copied successfully. File exists: {os.path.exists(cuda_file)}")
        else:
            print(f"Working CUDA file not found at {working_cuda_file}")
            print(f"Available files in /src/: {os.listdir('/src') if os.path.exists('/src') else 'Directory not found'}")
        
        # Set environment variables for compilation
        env = os.environ.copy()
        env["FORCE_CUDA"] = "1"
        env["CUDA_HOME"] = "/usr/local/cuda"
        
        # Run the compilation
        print("Starting CUDA compilation...")
        result = subprocess.run(
            ["python3", "setup.py", "build", "install"],
            cwd=ops_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"MultiScaleDeformableAttention CUDA op compilation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        else:
            print("MultiScaleDeformableAttention CUDA op compiled successfully")
        
        # Setup config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_common_config(cfg)
        add_swin_config(cfg)
        add_dinat_config(cfg)
        add_convnext_config(cfg)
        add_oneformer_config(cfg)
        
        # Use a default config - you may want to expose this as an input
        config_file = "/OneFormer/configs/ade20k/swin/oneformer_swin_large_bs16_160k.yaml"
        cfg.merge_from_file(config_file)
        
        # Set model weights path - you'll need to download these
        cfg.MODEL.WEIGHTS = "https://shi-labs.com/projects/oneformer/ade20k/swin_large_IN21k_384_bs16_160k/250_16_swin_l_oneformer_ade20k_160k.pth"
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.freeze()
        
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        
        # Setup metadata
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
        )
        self.cpu_device = torch.device("cpu")

    def predict(
        self,
        image: File = Input(description="Input image"),
        task: str = Input(
            default="panoptic",
            choices=["panoptic", "instance", "semantic"],
            description="Segmentation task type"
        ),
        output_type: str = Input(
            default="all", 
            choices=["all", "panoptic", "instance", "semantic"],
            description="Which outputs to return"
        )
    ) -> List[Path]:
        """Run segmentation on input image"""
        
        # Read input image
        input_path = str(image)
        img = read_image(input_path, format="BGR")
        
        # Convert to RGB for visualization
        img_rgb = img[:, :, ::-1]
        
        # Run prediction
        predictions = self.predictor(img, task)
        
        outputs = []
        
        # Generate visualizations based on task and output_type
        if task == 'panoptic' and (output_type == 'all' or output_type == 'panoptic'):
            visualizer = Visualizer(img_rgb, metadata=self.metadata, instance_mode=ColorMode.IMAGE)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info, alpha=0.7
            )
            
            # Save panoptic output
            panoptic_path = "/tmp/panoptic_output.png"
            vis_output.save(panoptic_path)
            outputs.append(Path(panoptic_path))

        if (task == 'panoptic' or task == 'semantic') and (output_type == 'all' or output_type == 'semantic'):
            visualizer = Visualizer(img_rgb, metadata=self.metadata, instance_mode=ColorMode.IMAGE_BW)
            if task == 'semantic':
                predictions = self.predictor(img, task)
            vis_output = visualizer.draw_sem_seg(
                predictions["sem_seg"].argmax(dim=0).to(self.cpu_device), alpha=0.7
            )
            
            # Save semantic output
            semantic_path = "/tmp/semantic_output.png"
            vis_output.save(semantic_path)
            outputs.append(Path(semantic_path))

        if (task == 'panoptic' or task == 'instance') and (output_type == 'all' or output_type == 'instance'):
            visualizer = Visualizer(img_rgb, metadata=self.metadata, instance_mode=ColorMode.IMAGE_BW)
            if task == 'instance':
                predictions = self.predictor(img, task)
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances, alpha=1)
            
            # Save instance output
            instance_path = "/tmp/instance_output.png"
            vis_output.save(instance_path)
            outputs.append(Path(instance_path))
        
        return outputs