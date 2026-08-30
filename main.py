import os

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    os.getenv("MAX_INPUT_TOKENS", "8192")
)

MAX_OUTPUT_TOKENS = int(
    os.getenv("MAX_OUTPUT_TOKENS", "512")
)

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


# ============================================================
# Request / Response
# ============================================================

class GenerateRequest(BaseModel):

    prompt: str = Field(
        ...,
        min_length=1
    )

    system_prompt: str | None = None

    max_output_tokens: int | None = Field(
        default=None,
        ge=1
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0
    )

    top_p: float = Field(
        default=0.95,
        gt=0.0,
        le=1.0
    )

    top_k: int = Field(
        default=64,
        ge=0
    )

    do_sample: bool = True


class GenerateResponse(BaseModel):

    model: str

    response: str

    input_tokens: int

    output_tokens: int


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

        global MODEL_NAME
        global MODEL_PATH

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
        # Clear GPU cache
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
        # 4-bit quantization
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
        # Model
        # ----------------------------------------------------

        print()
        print(
            "Loading Gemma model..."
        )

        print(
            "The model will be downloaded automatically "
            "if it is not already present."
        )

        self.model = AutoModelForMultimodalLM.from_pretrained(

            MODEL_NAME,

            quantization_config=bnb_config,

            device_map="auto",

            cache_dir=MODEL_PATH,

            torch_dtype=torch.bfloat16,

            low_cpu_mem_usage=True
        )

        self.model.eval()

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

        return (
            self.model,
            self.processor
        )


# ============================================================
# Load model ONCE
# ============================================================

model_loader = ModelLoader()

model, processor = model_loader.load()


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Gemma 4 12B Inference Server",
    version="1.0.0"
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

    return {

        "status": "ok",

        "model": MODEL_NAME,

        "cuda": torch.cuda.is_available(),

        "max_input_tokens":
            MAX_INPUT_TOKENS,

        "max_output_tokens":
            MAX_OUTPUT_TOKENS
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
# Generate
# ============================================================

@app.post(
    "/generate",
    response_model=GenerateResponse
)
def generate(
    request: GenerateRequest
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model is not loaded."
        )

    # --------------------------------------------------------
    # Output token limit
    # --------------------------------------------------------

    requested_output = (

        request.max_output_tokens

        if request.max_output_tokens is not None

        else MAX_OUTPUT_TOKENS
    )

    output_limit = min(
        requested_output,
        MAX_OUTPUT_TOKENS
    )

    # --------------------------------------------------------
    # Messages
    # --------------------------------------------------------

    messages = []

    if request.system_prompt:

        messages.append({

            "role": "system",

            "content":
                request.system_prompt
        })

    messages.append({

        "role": "user",

        "content": [

            {
                "type": "text",

                "text":
                    request.prompt
            }

        ]
    })

    try:

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

        inputs = processor.apply_chat_template(

            messages,

            tokenize=True,

            return_dict=True,

            return_tensors="pt",

            add_generation_prompt=True
        )

        input_tokens = (
            inputs["input_ids"]
            .shape[-1]
        )

        print(
            f"Input tokens: {input_tokens}"
        )

        # ----------------------------------------------------
        # Input limit
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Move tensors to GPU
        # ----------------------------------------------------

        device = next(
            model.parameters()
        ).device

        inputs = inputs.to(device)

        # ----------------------------------------------------
        # Generation parameters
        # ----------------------------------------------------

        generation_args = {

            "max_new_tokens":
                output_limit,

            "do_sample":
                request.do_sample,

            "top_p":
                request.top_p,

            "top_k":
                request.top_k
        }

        if request.do_sample:

            generation_args[
                "temperature"
            ] = request.temperature

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        print(
            f"Generating up to {output_limit} tokens..."
        )

        with torch.inference_mode():

            outputs = model.generate(

                **inputs,

                **generation_args
            )

        # ----------------------------------------------------
        # Remove prompt tokens
        # ----------------------------------------------------

        generated_ids = outputs[
            0,
            input_tokens:
        ]

        output_tokens = (
            generated_ids.shape[-1]
        )

        response = processor.decode(

            generated_ids,

            skip_special_tokens=True
        ).strip()

        print(
            f"Output tokens: {output_tokens}"
        )

        return GenerateResponse(

            model=MODEL_NAME,

            response=response,

            input_tokens=input_tokens,

            output_tokens=output_tokens
        )

    except HTTPException:

        raise

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()

        raise HTTPException(

            status_code=507,

            detail=(
                "GPU out of memory. "
                "Reduce input/output token limits."
            )
        )

    except Exception as exc:

        print(
            "Generation error:",
            exc
        )

        raise HTTPException(

            status_code=500,

            detail=str(exc)
        )


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000
    )