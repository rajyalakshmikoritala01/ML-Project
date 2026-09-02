import os
import shutil
import requests
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model import CottonDiseaseModel
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL")
HF_API_URL = (
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
    if HF_MODEL
    else None
)
headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
FALLBACK_UPLOAD_DIR = Path("/tmp/cotton_uploads")
MODEL_PATH = BASE_DIR / "model.h5"

UPLOAD_DIR.mkdir(exist_ok=True)

# -------- Lifespan --------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_service = CottonDiseaseModel(MODEL_PATH)
    print("✅ Model loaded successfully")
    yield
    print("🛑 Shutting down...")

app = FastAPI(
    title="Cotton Leaf Disease Detection API",
    version="2.0.0",
    description="AI Powered Cotton Disease Detection System",
    lifespan=lifespan
)

# -------- Static & Templates --------
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


CARE_GUIDE = {
    "Bacterial Blight": {
        "plant_part": "Leaf",
        "health_status": "Diseased",
        "summary": (
            "The model detected a diseased cotton leaf. Early containment is "
            "important to avoid spread and yield loss."
        ),
        "immediate_actions": [
            "Inspect nearby plants and tag affected rows for monitoring.",
            "Remove severely affected leaves and destroy them away from the field.",
            "Avoid overhead irrigation until symptoms stabilize.",
            "Keep workers/tools sanitized between infected and healthy zones.",
            "Apply approved crop-protection products based on local extension advice.",
        ],
        "prevention_actions": [
            "Follow field sanitation and weed control to reduce disease pressure.",
            "Treat seeds before sowing to prevent infection.",
            "Use balanced nutrition, especially potassium and micronutrients.",
            "Avoid excess nitrogen that promotes soft, disease-prone tissue.",
            "Apply bactericides such as Copper Oxychloride or Copper Hydroxide.",
        ],
        "yield_actions": [
            "Protect healthy upper canopy to retain photosynthesis.",
            "Maintain uniform soil moisture to reduce stress.",
            "Prioritize timely pest and disease monitoring after rain events.",
            "Record symptom progression plot-wise for targeted intervention.",
        ],
    },
    "Fusarium Wilt": {
        "plant_part": "Plant",
        "health_status": "Diseased",
        "summary": (
            "The model detected a diseased cotton plant. Focus on containment, "
            "root-zone health, and whole-plant stress reduction."
        ),
        "immediate_actions": [
            "Isolate severely affected plants where possible.",
            "Check stem/root zone for rot, wilt, or insect damage.",
            "Adjust irrigation to prevent prolonged waterlogging.",
            "Remove heavily infected plant debris from the field.",
            "Apply disease management inputs only as per label/local advisory.",
        ],
        "prevention_actions": [
            "Use Fusarium wilt-resistant cotton varieties.",
            "Rotate crops and avoid continuous cotton in high-pressure fields.",
            "Maintain well-drained soil conditions",
            "Avoid continuous cotton cultivation in the same field.",
            "Treat seeds with fungicides like Carbendazim.",
        ],
        "yield_actions": [
            "Support surviving plants with balanced fertilization split doses.",
            "Reduce additional stress from irrigation and nutrient shocks.",
            "Prioritize field sections with moderate infection for recovery.",
            "Track canopy vigor weekly to guide follow-up management.",
        ],
    },
    "Alternaria Leaf Spot":{
        "plant_part": "Plant",
        "health_status": "Diseased",
        "summary": (
            "The model detected a diseased cotton plant. Focus on containment, "
            "root-zone health, and whole-plant stress reduction."
        ),
        "immediate_actions": [
            "Isolate severely affected plants where possible.",
            "Check stem/root zone for rot, wilt, or insect damage.",
            "Adjust irrigation to prevent prolonged waterlogging.",
            "Remove heavily infected plant debris from the field.",
            "Apply disease management inputs only as per label/local advisory.",
        ],
        "prevention_actions": [
            "Use certified disease-free cotton seeds.",
            "Maintain proper spacing between plants for better air circulation.",
            "Remove and destroy infected leaves and crop residues.",
            "Avoid excessive irrigation and prolonged leaf wetness.",
            "Spray fungicides like Mancozeb or Chlorothalonil when symptoms appear.",
        ],
        "yield_actions": [
            "Support surviving plants with balanced fertilization split doses.",
            "Reduce additional stress from irrigation and nutrient shocks.",
            "Prioritize field sections with moderate infection for recovery.",
            "Track canopy vigor weekly to guide follow-up management.",
        ],
    },
    "Verticillium Wilt": {
        "plant_part": "Plant",
        "health_status": "Diseased",
        "summary": (
            "The model detected a diseased cotton plant. Focus on containment, "
            "root-zone health, and whole-plant stress reduction."
        ),
        "immediate_actions": [
            "Isolate severely affected plants where possible.",
            "Check stem/root zone for rot, wilt, or insect damage.",
            "Adjust irrigation to prevent prolonged waterlogging.",
            "Remove heavily infected plant debris from the field.",
            "Apply disease management inputs only as per label/local advisory.",
        ],
        "prevention_actions": [
            "Plant Verticillium wilt-resistant cotton varieties.",
            "Practice crop rotation with crops like maize or wheat.",
            "Improve soil fertility with organic manure.",
            "Remove infected plants to prevent spread.",
            "Apply fungicides such as Thiophanate-methyl or biological control like Trichoderma harzianum.",
        ],
        "yield_actions": [
            "Support surviving plants with balanced fertilization split doses.",
            "Reduce additional stress from irrigation and nutrient shocks.",
            "Prioritize field sections with moderate infection for recovery.",
            "Track canopy vigor weekly to guide follow-up management.",
        ],

    },
    "Healthy Cotton Leaf": {
        "plant_part": "Leaf",
        "health_status": "Fresh",
        "summary": (
            "The model detected a fresh cotton leaf. Keep prevention practices "
            "strong to sustain high productivity."
        ),
        "immediate_actions": [
            "No major disease sign detected in this image.",
            "Continue routine scouting and photo logging.",
            "Keep irrigation and fertilizer schedule consistent.",
        ],
        "prevention_actions": [
            "Maintain clean field borders and weed control.",
            "Use preventive IPM practices before peak humidity.",
            "Avoid prolonged leaf wetness from late-day irrigation.",
            "Scout underside of leaves for early pest activity.",
            "Maintain balanced nutrition to preserve leaf strength.",
        ],
        "yield_actions": [
            "Optimize nitrogen-potassium balance for boll development.",
            "Maintain uniform plant population and canopy light penetration.",
            "Schedule irrigation by soil moisture, not calendar only.",
            "Use periodic leaf/tissue testing for nutrient corrections.",
            "Track growth stage and align inputs to critical crop windows.",
        ],
    },
    "Healthy Cotton Plant": {
        "plant_part": "Plant",
        "health_status": "Fresh",
        "summary": (
            "The model detected a fresh cotton plant. Focus on sustaining plant "
            "vigor and protecting against future disease pressure."
        ),
        "immediate_actions": [
            "No visible disease signal in this plant image.",
            "Keep regular monitoring frequency unchanged.",
            "Preserve current irrigation and nutrient discipline.",
        ],
        "prevention_actions": [
            "Use clean cultivation and crop-residue management.",
            "Prevent moisture stress swings that weaken plants.",
            "Monitor lower canopy and root zone for early stress signs.",
            "Rotate chemistry classes if preventive sprays are used.",
            "Keep pest populations below threshold to avoid disease entry points.",
        ],
        "yield_actions": [
            "Support flowering/boll set with stage-wise nutrient planning.",
            "Use split fertilizer application to improve uptake efficiency.",
            "Protect square and boll retention through stress management.",
            "Maintain optimal irrigation intervals by soil and weather.",
            "Audit field variability and correct weak zones early.",
        ],
    },
}


def _fallback_ai_report(
    prediction: str, summary: str, prevention_actions: List[str], yield_actions: List[str]
) -> str:
    prevention = "; ".join(prevention_actions[:3])
    yield_plan = "; ".join(yield_actions[:3])
    return (
        f"Condition: {prediction}. {summary} "
        f"Prevention focus: {prevention}. "
        f"Yield focus: {yield_plan}."
    )


def build_crop_analysis(prediction: str, probabilities: Dict[str, float]):
    guide = CARE_GUIDE.get(
        prediction,
        {
            "plant_part": "Unknown",
            "health_status": "Unknown",
            "summary": "Model output class is not mapped to a guidance profile.",
            "immediate_actions": ["Review the image and run prediction again."],
            "prevention_actions": ["Use standard field hygiene and scouting."],
            "yield_actions": ["Maintain balanced nutrition and irrigation."],
        },
    )

    top_class = prediction
    top_score = probabilities.get(prediction, 0.0)
    confidence_pct = round(top_score * 100.0, 2)

    if top_score >= 0.80:
        confidence_band = "High"
    elif top_score >= 0.60:
        confidence_band = "Medium"
    else:
        confidence_band = "Low"

    probability_ranking = [
        {
            "class_name": class_name,
            "probability": round(probability, 6),
            "percentage": round(probability * 100.0, 2),
        }
        for class_name, probability in probabilities.items()
    ]
    probability_ranking.sort(key=lambda x: x["probability"], reverse=True)

    notes = [
        "This is an image-based model prediction, not a lab diagnosis.",
        "Use local agronomist/extension guidance for final treatment decisions.",
    ]
    if confidence_band == "Low":
        notes.append(
            "Prediction confidence is low. Capture a clearer image and re-check."
        )
    if top_class != prediction:
        notes.append("Top probability class does not match selected prediction label.")

    analysis = {
        "plant_part": guide["plant_part"],
        "health_status": guide["health_status"],
        "summary": guide["summary"],
        "confidence_percent": confidence_pct,
        "confidence_band": confidence_band,
    }
    recommendations = {
        "immediate_actions": guide["immediate_actions"],
        "prevention_actions": guide["prevention_actions"],
        "yield_actions": guide["yield_actions"],
    }

    return analysis, recommendations, probability_ranking, notes


def generate_ai_report(
    disease_name: str, summary: str, prevention_actions: List[str], yield_actions: List[str]
):
    if not HF_API_KEY or not HF_API_URL:
        return _fallback_ai_report(
            disease_name, summary, prevention_actions, yield_actions
        )

    prompt = f"""
    Cotton crop condition: {disease_name}

    Explain:
    1. Disease explanation
    2. Causes
    3. Prevention methods
    4. Recommended pesticides
    5. Fertilizer suggestions
    6. Farmer advice.

    Keep answer simple and under 180 words.
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.5
        },
        "options": {
            "wait_for_model": True
        }
    }

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return _fallback_ai_report(
                disease_name, summary, prevention_actions, yield_actions
            )

        result = response.json()

        if isinstance(result, dict) and "error" in result:
            return _fallback_ai_report(
                disease_name, summary, prevention_actions, yield_actions
            )

        return result[0]["generated_text"]

    except Exception:
        return _fallback_ai_report(
            disease_name, summary, prevention_actions, yield_actions
        )


# -------- UI Route --------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# -------- Prediction API --------
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are supported"
        )

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except PermissionError:
        FALLBACK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_path = FALLBACK_UPLOAD_DIR / safe_filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Model prediction (make sure your model.py returns prediction & probabilities)
    prediction, probabilities = app.state.model_service.predict(file_path)
    analysis, recommendations, probability_ranking, notes = build_crop_analysis(
        prediction, probabilities
    )

    ai_report = generate_ai_report(
        prediction,
        analysis["summary"],
        recommendations["prevention_actions"],
        recommendations["yield_actions"],
    )

    return {
        "filename": safe_filename,
        "prediction": prediction,
        "confidence_percent": analysis["confidence_percent"],
        "confidence_band": analysis["confidence_band"],
        "probabilities": probabilities,
        "probability_ranking": probability_ranking,
        "crop_analysis": analysis,
        "recommendations": recommendations,
        "model_notes": notes,
        "ai_agronomist_report": ai_report
    }