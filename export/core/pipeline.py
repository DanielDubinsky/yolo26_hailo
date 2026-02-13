import logging
import datetime
from pathlib import Path
from typing import List, Optional
from ..config import ExportConfig, get_variant_config
from .logger import setup_logger, redirect_output_to_log, restore_output
from ..steps.base import Step
from ..steps.extract import ExtractSubgraphsStep
from ..steps.convert import OnnxToHarStep
from ..steps.quantize import QuantizeStep
from ..steps.compile import CompileStep

class ExportPipeline:
    def __init__(self, config: ExportConfig):
        self.config = config
        self.logger = logging.getLogger("export.pipeline")
        self.steps: List[Step] = []
        
        # Initialize steps
        self.steps.append(ExtractSubgraphsStep(config))
        self.steps.append(OnnxToHarStep(config))
        self.steps.append(QuantizeStep(config))
        self.steps.append(CompileStep(config))

    def _create_run_structure(self) -> Path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Format: experiments/{variant}_{target}_{timestamp} or {variant}_{target}_{tag}_{timestamp}
        run_name = f"{self.config.variant}_{self.config.target}"
        if self.config.tag:
            run_name += f"_{self.config.tag}"
        run_name += f"_{timestamp}"
        
        # Determine experiments directory 
        # Assuming run from repo root or export dir, let's anchor to repo root
        # If we are in /.../yolo26_hailo/export/core/pipeline.py
        # Repo root is /.../yolo26_hailo
        repo_root = Path(__file__).parent.parent.parent
        experiments_dir = repo_root / "experiments"
        
        run_dir = experiments_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run(self):
        # 1. Setup Run Directory
        run_dir = self._create_run_structure()
        self.config.output_dir = run_dir
        
        # 2. Setup Logger
        log_file = run_dir / "run.log"
        setup_logger("export", log_file) 
        
        # 3. Redirect stdout and stderr to log file
        redirectors = redirect_output_to_log(log_file)
        
        try:
            self.logger.info(f"Starting export run for {self.config.variant} on {self.config.target}")
            self.logger.info(f"Run directory: {run_dir}")
            self.logger.info(f"Input ONNX: {self.config.onnx_path}")
            
            # 4. Context Initialization
            context = {
                'config': self.config,
                'variant_config': get_variant_config(self.config.variant),
                'run_dir': run_dir
            }
            
            # 5. Execute Steps
            for step in self.steps:
                context = step.run(context)
                
            self.logger.info("--- Export Process Completed Successfully ---")
            self.logger.info(f"HEF File: {context.get('hef_path')}")
            
        except Exception as e:
            self.logger.error(f"Export failed: {e}", exc_info=True)
            raise
        finally:
            # Restore original stdout/stderr
            restore_output(*redirectors)

