import os
import io

import torch
import librosa

from PIL import Image

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
)

from pydantic import BaseModel

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "google/gemma-4-12B-it"
)

MAX_INPUT_TOKENS = int(
    os.getenv(
        "MAX_INPUT_TOKENS",
        "8192"
    )
)

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "MAX_OUTPUT_TOKENS",
        "512"
    )
)


# ============================================================
# Local Model Directory
# ============================================================
#
# The model is downloaded into this deployment-specific folder.
#
# No global HuggingFace cache is configured.
#
# If the Kubernetes Job is deleted, the downloaded model is
# deleted with the container.
#
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "gemma-4-12b"
)

os.makedirs(
    MODEL_PATH,
    exist_ok=True
)


# ============================================================
# Globals
# ============================================================

model = None
processor = None

_dtype_hooks = []


# ============================================================
# Response Model
# ============================================================

class GenerateResponse(BaseModel):

    model: str

    response: str

    input_tokens: int

    output_tokens: int


# ============================================================
# Multimodal Dtype Helpers
# ============================================================

MULTIMODAL_KEYWORDS = (
    "vision",
    "visual",
    "image",
    "audio",
    "multimodal",
    "multi_modal",
    "mm_",
    "mm.",
    "embedder",
    "projector",
    "projection",
)


def is_multimodal_module(name: str) -> bool:

    name = name.lower()

    return any(
        keyword in name
        for keyword in MULTIMODAL_KEYWORDS
    )


def get_module_parameter_dtype(module):

    """
    Return the first floating-point parameter dtype of a module.

    Quantized BitsAndBytes modules are intentionally ignored.

    We only use this for the multimodal dtype-alignment hooks.
    """

    for parameter in module.parameters(
        recurse=False
    ):

        if parameter.is_floating_point():

            return parameter.dtype

    return None


def align_multimodal_inputs(module, inputs):

    """
    Align floating-point inputs with the dtype expected by the
    receiving multimodal module.

    This does NOT blindly convert all tensors.

    Integer tensors such as input_ids and position IDs remain
    untouched.

    Only floating-point tensors are considered.
    """

    target_dtype = get_module_parameter_dtype(
        module
    )

    if target_dtype is None:

        return inputs

    def convert(value):

        if isinstance(value, torch.Tensor):

            if value.is_floating_point():

                if value.dtype != target_dtype:

                    return value.to(
                        dtype=target_dtype
                    )

            return value

        if isinstance(value, tuple):

            return tuple(
                convert(item)
                for item in value
            )

        if isinstance(value, list):

            return [
                convert(item)
                for item in value
            ]

        return value

    return tuple(
        convert(value)
        for value in inputs
    )


def multimodal_dtype_hook(
    module,
    inputs
):

    return align_multimodal_inputs(
        module,
        inputs
    )


def install_multimodal_dtype_hooks(model):

    """
    Install lightweight forward-pre-hooks only on multimodal
    modules that have floating-point parameters.

    This is deliberately narrow.

    We do NOT:
        - convert the whole model to FP32
        - convert the whole model to BF16
        - modify 4-bit BitsAndBytes layers
        - modify token IDs or attention masks
    """

    global _dtype_hooks

    installed = 0

    print()
    print("==========================================")
    print(" Multimodal Dtype Compatibility Setup")
    print("==========================================")

    for name, module in model.named_modules():

        if not is_multimodal_module(name):

            continue

        # Don't interfere with BitsAndBytes quantized modules.
        class_name = module.__class__.__name__.lower()

        if "linear4bit" in class_name:
            continue

        if "linear8bit" in class_name:
            continue

        dtype = get_module_parameter_dtype(
            module
        )

        if dtype is None:

            continue

        handle = module.register_forward_pre_hook(
            multimodal_dtype_hook
        )

        _dtype_hooks.append(
            handle
        )

        installed += 1

        print(
            f"[dtype-hook] {name}"
            f" -> {dtype}"
        )

    print()
    print(
        f"Installed {installed} multimodal dtype hooks."
    )

    if installed == 0:

        print(
            "No matching multimodal modules found."
        )


def remove_multimodal_dtype_hooks():

    global _dtype_hooks

    for handle in _dtype_hooks:

        try:

            handle.remove()

        except Exception:

            pass

    _dtype_hooks = []


# ============================================================
# Model Loader
# ============================================================

class ModelLoader:

    def __init__(self):

        self.model = None
        self.processor = None

        print(
            "CUDA available:",
            torch.cuda.is_available()
        )

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

            print(
                "GPU memory:",
                round(
                    torch.cuda.get_device_properties(0).total_memory
                    / (1024 ** 3),
                    2
                ),
                "GB"
            )

    def load(self):

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is not available."
            )

        print()
        print("==========================================")
        print("        Gemma 4 12B Model Loader")
        print("==========================================")
        print()

        print(
            "Model:",
            MODEL_NAME
        )

        print(
            "Model path:",
            MODEL_PATH
        )

        print(
            "Max input tokens:",
            MAX_INPUT_TOKENS
        )

        print(
            "Max output tokens:",
            MAX_OUTPUT_TOKENS
        )

        # ----------------------------------------------------
        # CUDA information
        # ----------------------------------------------------

        print()
        print("CUDA information:")

        print(
            "  PyTorch:",
            torch.__version__
        )

        print(
            "  CUDA:",
            torch.version.cuda
        )

        print(
            "  Device:",
            torch.cuda.get_device_name(0)
        )

        # ----------------------------------------------------
        # Clear CUDA cache
        # ----------------------------------------------------

        print()
        print("Clearing CUDA cache...")

        torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Processor
        # ----------------------------------------------------

        print()
        print("Loading processor...")

        self.processor = AutoProcessor.from_pretrained(

            MODEL_NAME,

            cache_dir=MODEL_PATH
        )

        print(
            "Processor loaded."
        )

        # ----------------------------------------------------
        # 4-bit NF4 Quantization
        # ----------------------------------------------------

        print()
        print(
            "Configuring 4-bit NF4 quantization..."
        )

        bnb_config = BitsAndBytesConfig(

            load_in_4bit=True,

            bnb_4bit_compute_dtype=torch.bfloat16,

            bnb_4bit_use_double_quant=True,

            bnb_4bit_quant_type="nf4"
        )

        # ----------------------------------------------------
        # Load Model
        # ----------------------------------------------------

        print()
        print(
            "Loading Gemma model..."
        )

        print(
            "The model will be downloaded automatically "
            "if it is not already present."
        )

        #
        # IMPORTANT:
        #
        # Do not force dtype=torch.bfloat16 here.
        #
        # The model contains multimodal components with their
        # own dtype requirements.
        #
        # BitsAndBytes handles the 4-bit language-model weights.
        #
        # ----------------------------------------------------

        self.model = AutoModelForMultimodalLM.from_pretrained(

            MODEL_NAME,

            quantization_config=bnb_config,

            device_map="auto",

            cache_dir=MODEL_PATH,

            low_cpu_mem_usage=True
        )

        self.model.eval()

        # ----------------------------------------------------
        # Install targeted multimodal dtype compatibility hooks
        # ----------------------------------------------------

        install_multimodal_dtype_hooks(
            self.model
        )

        # ----------------------------------------------------
        # Model information
        # ----------------------------------------------------

        print()
        print("==========================================")
        print("       Gemma 4 12B Model Ready")
        print("==========================================")

        print(
            "Device map:",
            getattr(
                self.model,
                "hf_device_map",
                "unknown"
            )
        )

        # ----------------------------------------------------
        # GPU memory after model load
        # ----------------------------------------------------

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated(0)
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved(0)
                / (1024 ** 3)
            )

            print()
            print(
                f"GPU memory allocated: "
                f"{allocated:.2f} GB"
            )

            print(
                f"GPU memory reserved: "
                f"{reserved:.2f} GB"
            )

        return (
            self.model,
            self.processor
        )


# ============================================================
# Load Model ONCE
# ============================================================

model_loader = ModelLoader()

model, processor = model_loader.load()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(

    title="Gemma 4 12B Inference Server",

    description=(
        "Multimodal Gemma 4 12B inference API "
        "supporting text, optional images, and optional audio."
    ),

    version="1.3.0"
)


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    if model is None:

        raise HTTPException(

            status_code=503,

            detail="Model is not loaded."
        )

    gpu_memory = None

    if torch.cuda.is_available():

        gpu_memory = {

            "allocated_gb": round(
                torch.cuda.memory_allocated(0)
                / (1024 ** 3),
                2
            ),

            "reserved_gb": round(
                torch.cuda.memory_reserved(0)
                / (1024 ** 3),
                2
            ),

            "total_gb": round(
                torch.cuda.get_device_properties(0)
                .total_memory
                / (1024 ** 3),
                2
            )
        }

    return {

        "status":
            "ok",

        "model":
            MODEL_NAME,

        "cuda":
            torch.cuda.is_available(),

        "torch_version":
            torch.__version__,

        "cuda_version":
            torch.version.cuda,

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "gpu_memory":
            gpu_memory,

        "max_input_tokens":
            MAX_INPUT_TOKENS,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "modalities": [

            "text",

            "image",

            "audio"
        ]
    }


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "Gemma 4 12B Inference Server",

        "model":
            MODEL_NAME,

        "max_input_tokens":
            MAX_INPUT_TOKENS,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS,

        "modalities": [

            "text",

            "image",

            "audio"
        ],

        "endpoints": {

            "health":
                "/health",

            "generate":
                "/generate",

            "docs":
                "/docs"
        }
    }


# ============================================================
# Move Input Tensors to CUDA
# ============================================================

def move_inputs_to_cuda(inputs):

    """
    Move tensors to CUDA while preserving their dtype.

    IMPORTANT:

    We do NOT call:

        inputs.to(torch.bfloat16)

    because:
        input_ids must remain integer
        attention masks must remain integer/bool
        image/audio tensors may require different dtypes
    """

    moved = {}

    for key, value in inputs.items():

        if isinstance(
            value,
            torch.Tensor
        ):

            moved[key] = value.to(

                "cuda",

                non_blocking=True
            )

        else:

            moved[key] = value

    return moved


# ============================================================
# Generate
# ============================================================

@app.post(
    "/generate",
    response_model=GenerateResponse
)
async def generate(

    prompt: str = Form(...),

    system_prompt: str | None = Form(
        default=None
    ),

    image: UploadFile | None = File(
        default=None
    ),

    audio: UploadFile | None = File(
        default=None
    ),

    max_output_tokens: int | None = Form(
        default=None
    ),

    temperature: float = Form(
        default=0.7
    ),

    top_p: float = Form(
        default=0.95
    ),

    top_k: int = Form(
        default=64
    ),

    do_sample: bool = Form(
        default=True
    )
):

    # ========================================================
    # Validate model
    # ========================================================

    if model is None:

        raise HTTPException(

            status_code=503,

            detail="Model is not loaded."
        )

    # ========================================================
    # Validate generation parameters
    # ========================================================

    if temperature < 0.0 or temperature > 2.0:

        raise HTTPException(

            status_code=400,

            detail=(
                "temperature must be between "
                "0.0 and 2.0."
            )
        )

    if top_p <= 0.0 or top_p > 1.0:

        raise HTTPException(

            status_code=400,

            detail=(
                "top_p must be greater than 0 "
                "and at most 1.0."
            )
        )

    if top_k < 0:

        raise HTTPException(

            status_code=400,

            detail="top_k cannot be negative."
        )

    # ========================================================
    # Output token limit
    # ========================================================

    requested_output = (

        max_output_tokens

        if max_output_tokens is not None

        else MAX_OUTPUT_TOKENS
    )

    if requested_output < 1:

        raise HTTPException(

            status_code=400,

            detail=(
                "max_output_tokens must be "
                "at least 1."
            )
        )

    output_limit = min(

        requested_output,

        MAX_OUTPUT_TOKENS
    )

    # ========================================================
    # Build multimodal content
    # ========================================================

    content = []

    # ========================================================
    # IMAGE
    # ========================================================

    if image is not None:

        try:

            print(
                f"Receiving image: {image.filename}"
            )

            image_bytes = await image.read()

            if not image_bytes:

                raise ValueError(
                    "Uploaded image is empty."
                )

            pil_image = Image.open(

                io.BytesIO(
                    image_bytes
                )

            ).convert("RGB")

            content.append({

                "type":
                    "image",

                "image":
                    pil_image
            })

            print(
                "Image loaded successfully."
            )

        except Exception as exc:

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Invalid image: {exc}"
                )
            )

    # ========================================================
    # AUDIO
    # ========================================================

    if audio is not None:

        try:

            print(
                f"Receiving audio: {audio.filename}"
            )

            audio_bytes = await audio.read()

            if not audio_bytes:

                raise ValueError(
                    "Uploaded audio is empty."
                )

            audio_data, sample_rate = librosa.load(

                io.BytesIO(
                    audio_bytes
                ),

                sr=16000,

                mono=True
            )

            content.append({

                "type":
                    "audio",

                "audio":
                    audio_data
            })

            print(
                f"Audio loaded successfully. "
                f"Sample rate: {sample_rate}, "
                f"Samples: {len(audio_data)}"
            )

        except Exception as exc:

            raise HTTPException(

                status_code=400,

                detail=(
                    f"Invalid audio: {exc}"
                )
            )

    # ========================================================
    # TEXT
    # ========================================================

    content.append({

        "type":
            "text",

        "text":
            prompt
    })

    # ========================================================
    # Messages
    # ========================================================

    messages = []

    if system_prompt:

        messages.append({

            "role":
                "system",

            "content":
                system_prompt
        })

    messages.append({

        "role":
            "user",

        "content":
            content
    })

    try:

        # ====================================================
        # Processor
        # ====================================================

        print()
        print(
            "Processing multimodal input..."
        )

        inputs = processor.apply_chat_template(

            messages,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",

            add_generation_prompt=True
        )

        # ====================================================
        # Print input tensor information
        # ====================================================

        print(
            "Input tensors:"
        )

        for key, value in inputs.items():

            if isinstance(
                value,
                torch.Tensor
            ):

                print(

                    f"  {key}: "
                    f"shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}, "
                    f"device={value.device}"
                )

        # ====================================================
        # Input token count
        # ====================================================

        input_tokens = (

            inputs["input_ids"]
            .shape[-1]
        )

        print(
            f"Input tokens: {input_tokens}"
        )

        # ====================================================
        # Input token limit
        # ====================================================

        if input_tokens > MAX_INPUT_TOKENS:

            raise HTTPException(

                status_code=413,

                detail={

                    "error":
                        "input_too_long",

                    "input_tokens":
                        input_tokens,

                    "max_input_tokens":
                        MAX_INPUT_TOKENS
                }
            )

        # ====================================================
        # Move inputs to CUDA
        # ====================================================

        print(
            "Moving input tensors to CUDA..."
        )

        inputs = move_inputs_to_cuda(
            inputs
        )

        # ====================================================
        # Print CUDA input information
        # ====================================================

        print(
            "CUDA input tensors:"
        )

        for key, value in inputs.items():

            if isinstance(
                value,
                torch.Tensor
            ):

                print(

                    f"  {key}: "
                    f"shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}, "
                    f"device={value.device}"
                )

        # ====================================================
        # Generation arguments
        # ====================================================

        generation_args = {

            "max_new_tokens":
                output_limit,

            "do_sample":
                do_sample
        }

        if do_sample:

            generation_args.update({

                "temperature":
                    temperature,

                "top_p":
                    top_p,

                "top_k":
                    top_k
            })

        # ====================================================
        # Generate
        # ====================================================

        print()
        print(
            f"Generating up to "
            f"{output_limit} tokens..."
        )

        with torch.inference_mode():

            outputs = model.generate(

                **inputs,

                **generation_args
            )

        # ====================================================
        # Remove input tokens
        # ====================================================

        generated_ids = outputs[

            0,

            input_tokens:
        ]

        output_tokens = (

            generated_ids.shape[-1]
        )

        print(
            f"Output tokens: {output_tokens}"
        )

        # ====================================================
        # Decode
        # ====================================================

        response = processor.decode(

            generated_ids,

            skip_special_tokens=True
        ).strip()

        print(
            "Generation complete."
        )

        # ====================================================
        # Response
        # ====================================================

        return GenerateResponse(

            model=MODEL_NAME,

            response=response,

            input_tokens=input_tokens,

            output_tokens=output_tokens
        )

    # ========================================================
    # HTTP Exceptions
    # ========================================================

    except HTTPException:

        raise

    # ========================================================
    # CUDA OOM
    # ========================================================

    except torch.cuda.OutOfMemoryError:

        print(
            "CUDA OUT OF MEMORY"
        )

        torch.cuda.empty_cache()

        raise HTTPException(

            status_code=507,

            detail={

                "error":
                    "gpu_out_of_memory",

                "message": (
                    "GPU out of memory. "
                    "Reduce input size or "
                    "max_output_tokens."
                )
            }
        )

    # ========================================================
    # Runtime Errors
    # ========================================================

    except RuntimeError as exc:

        print(
            "Runtime error:",
            repr(exc)
        )

        torch.cuda.empty_cache()

        raise HTTPException(

            status_code=500,

            detail=str(exc)
        )

    # ========================================================
    # General Errors
    # ========================================================

    except Exception as exc:

        print(
            "Generation error:",
            repr(exc)
        )

        raise HTTPException(

            status_code=500,

            detail=str(exc)
        )


# ============================================================
# Run Directly
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000
    )
