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
sys.path.append("/OneFormer/demo")

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

        # Copy the working visualizer file to the correct location
        visualizer_file = "/OneFormer/demo/visualizer.py"
        working_visualizer_file = "/src/visualizer.py"
        
        print(f"Checking for working visualizer file at: {working_visualizer_file}")
        if os.path.exists(working_visualizer_file):
            print(f"Found working visualizer file, copying to: {visualizer_file}")
            # Remove existing file if it exists
            if os.path.exists(visualizer_file):
                os.remove(visualizer_file)
            # Copy the working file
            shutil.copy(working_visualizer_file, visualizer_file)
            print(f"File copied successfully. File exists: {os.path.exists(visualizer_file)}")
        else:
            print(f"Working visualizer file not found at {working_visualizer_file}")
            print(f"Available files in /src/: {os.listdir('/src') if os.path.exists('/src') else 'Directory not found'}")
        
        # Set environment variables for compilation
        env = os.environ.copy()
        env["FORCE_CUDA"] = "1"
        env["CUDA_HOME"] = "/usr/local/cuda"
        
        # Run the compilation in place first
        print("Starting CUDA compilation...")
        result = subprocess.run(
            ["python3", "setup.py", "build_ext", "--inplace"],
            cwd=ops_dir,
            env=env,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            # Also try global installation for backup
            print("Installing globally as backup...")
            subprocess.run(
                ["python3", "setup.py", "build", "install"],
                cwd=ops_dir,
                env=env,
                capture_output=True,
                text=True
            )
        
        if result.returncode != 0:
            print(f"MultiScaleDeformableAttention CUDA op compilation failed:")
            print(f"Return code: {result.returncode}")
        else:
            print("MultiScaleDeformableAttention CUDA op compiled and installed successfully")
        
        # Test and debug import issues
        print("CUDA module installation completed, debugging import paths...")
        
        # Add the ops directory to Python path
        sys.path.insert(0, ops_dir)
        
        # Check what files were created
        print(f"Files in ops dir: {os.listdir(ops_dir)}")
        so_files = [f for f in os.listdir(ops_dir) if f.endswith('.so')]
        print(f"Found .so files: {so_files}")
        
        # Check build directory
        if os.path.exists(os.path.join(ops_dir, "build")):
            build_dir = os.path.join(ops_dir, "build")
            for root, dirs, files in os.walk(build_dir):
                so_files_build = [f for f in files if f.endswith('.so')]
                if so_files_build:
                    print(f"Found .so files in {root}: {so_files_build}")
                    sys.path.insert(0, root)
        
        # Try to test import the module
        try:
            import MultiScaleDeformableAttention as MSDA
            print("Successfully imported MultiScaleDeformableAttention!")
        except ImportError as e:
            print(f"Import still failing: {e}")
            print(f"Python path includes: {[p for p in sys.path if 'ops' in p]}")
            
            # Try importing from site-packages
            import site
            site_packages = site.getsitepackages()
            print(f"Site packages: {site_packages}")
            
            # List installed packages that might be relevant
            for path in site_packages:
                if os.path.exists(path):
                    relevant = [f for f in os.listdir(path) if 'MultiScale' in f or 'Deform' in f]
                    if relevant:
                        print(f"Found in {path}: {relevant}")
        
        print("Proceeding with OneFormer imports...")
        
        # Now import OneFormer modules after CUDA compilation
        from detectron2.config import get_cfg
        from detectron2.data.detection_utils import read_image
        from detectron2.projects.deeplab import add_deeplab_config
        from detectron2.data import MetadataCatalog

        from oneformer import (
            add_oneformer_config,
            add_common_config,
            add_swin_config,
            add_dinat_config,
            add_convnext_config,
        )

        from defaults import DefaultPredictor
        from visualizer import ColorMode, Visualizer
        
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
        
        # Set model weights path - using Hugging Face mirror
        cfg.MODEL.WEIGHTS = "https://huggingface.co/shi-labs/oneformer_ade20k_swin_large/resolve/main/250_16_swin_l_oneformer_ade20k_160k.pth"
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
        
        # Import read_image if not already available
        from detectron2.data.detection_utils import read_image
        from visualizer import ColorMode, Visualizer
        
        # Read input image - handle cog File input properly
        if hasattr(image, 'read'):
            # It's a file-like object, save it temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(image.read())
                input_path = temp_file.name
        else:
            # It's already a path
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