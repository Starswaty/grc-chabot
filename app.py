import cohere
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.Client(COHERE_API_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Control Validator API with Cohere is working."}

@app.post("/validate-control")
async def validate_control(
    process: str = Query(...),
    subprocess: str = Query(...),
    risk: str = Query(...),
    frequency: str = Query(...),
    risk_description: str = Query(...),
    control: str = Query(...),
    control_description: str = Query(...)
):
    try:
        prompt = f"""
        You are a smart internal control analyst.
        Given the following information:
        Process: {process}
        Subprocess: {subprocess}
        Risk: {risk}
        Frequency: {frequency}
        Risk Description: {risk_description}
        Control: {control}
        Control Description: {control_description}

        Validate if the control mitigates the risk properly. If yes, say VALID and explain why. If not, say INVALID and explain what's missing or could be improved.
        """

        response = co.generate(
            model='command-xlarge',
            prompt=prompt,
            max_tokens=300,
            temperature=0.7
        )

        return {"result": response.generations[0].text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
