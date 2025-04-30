import os
import sqlite3
import logging
import io
import csv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import settings
from data import SQLRepository
from model import GarchModel

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="GARCH Model API",
    description="Train GARCH models on stock returns and forecast volatility.",
    version="1.0"
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class FitIn(BaseModel):
    ticker: str
    use_new_data: bool
    n_observations: int
    p: int | None
    q: int | None

class FitOut(FitIn):
    success: bool
    message: str

class PredictIn(BaseModel):
    ticker: str
    n_days: int

class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/form", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

def build_model(ticker: str, use_new_data: bool) -> GarchModel:
    connection = sqlite3.connect(settings.db_name, check_same_thread=False)
    repo = SQLRepository(connection=connection)
    return GarchModel(ticker=ticker, use_new_data=use_new_data, repo=repo)

@app.post("/fit_form", response_class=HTMLResponse)
async def fit_form_ui(
    request: Request,
    ticker: str = Form(...),
    use_new_data: str = Form(...),
    n_observations: int = Form(...),
    mode: str = Form("manual"),
    p: int = Form(1),
    q: int = Form(1)
):
    try:
        use_auto = (mode == "auto")
        model = build_model(ticker, use_new_data.lower() == "true")
        model.wrangle_data(n_observations)

        if use_auto:
            p, q, bic = model.find_best_garch_params(range(1, 5), range(1, 5))
            model.train(p, q)
            message = f"Auto-selected GARCH({p},{q}) with BIC={bic:.2f}."
        else:
            model.train(p, q)
            message = f"Trained GARCH({p},{q}) model."

        filepath = model.dump()
        diagnostics = {
            "aic": model.aic,
            "bic": model.bic,
            "lb_pvalue": getattr(model, "lb_pvalue", None),
            "train_time": str(model.train_time),
            "n_obs": model.n_obs,
        }
        result = {"success": True, "message": message + f" Model saved at {filepath}", **diagnostics}
    except FileNotFoundError:
        result = {"success": False, "message": "Model not found. Please train a model first."}
    except ValueError as ve:
        result = {"success": False, "message": f"Input error: {ve}"}
    except Exception as e:
        result = {"success": False, "message": "An unexpected error occurred. Please try again or contact support."}

    return templates.TemplateResponse("form.html", {"request": request, "result": result})

@app.post("/download_csv", response_class=StreamingResponse)
def download_csv(ticker: str = Form(...), n_days: int = Form(...)):
    try:
        model = build_model(ticker, use_new_data=False)
        model.load()
        forecast = model.predict_volatility(n_days)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Date", "Forecast Volatility", "Lower CI", "Upper CI"])
        for date, mean in forecast["mean"].items():
            lower = forecast["lower"][date]
            upper = forecast["upper"][date]
            writer.writerow([str(date), mean, lower, upper])
        output.seek(0)
        return StreamingResponse(output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={ticker}_forecast.csv"})
    except Exception as e:
        return HTMLResponse(f"<h3>Error generating CSV: {str(e)}</h3>", status_code=500)

@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form_ui(
    request: Request,
    ticker: str = Form(...),
    n_days: int = Form(...)
):
    try:
        model = build_model(ticker, use_new_data=False)
        model.load()
        forecast = model.predict_volatility(n_days)
        result = {
            "success": True,
            "message": f"Forecast completed for {n_days} days.",
            "forecast": forecast,
            "ticker": ticker,
            "n_days": n_days
        }
    except FileNotFoundError:
        result = {"success": False, "message": "Model not found. Please train a model first."}
    except Exception as e:
        result = {"success": False, "message": "An unexpected error occurred. Please try again or contact support."}

    return templates.TemplateResponse("form.html", {"request": request, "result": result})

@app.post("/fit", response_model=FitOut, tags=["Model"])
async def fit_model_api(request: FitIn):
    response = request.dict()
    try:
        model = build_model(request.ticker, request.use_new_data)
        model.wrangle_data(request.n_observations)
        if request.p is None or request.q is None:
            best_p, best_q, bic = model.find_best_garch_params()
            model.train(best_p, best_q)
            response["message"] = f"Auto-selected parameters: p={best_p}, q={best_q}"
        else:
            model.train(request.p, request.q)
            response["message"] = f"Model trained with p={request.p}, q={request.q}"
        model.dump()
        response["success"] = True
    except Exception as e:
        response.update(success=False, message=str(e))
    return response

@app.post("/predict", response_model=PredictOut, tags=["Model"])
async def predict_volatility_api(request: PredictIn):
    response = request.dict()
    try:
        model = build_model(request.ticker, use_new_data=False)
        model.load()
        forecast = model.predict_volatility(request.n_days)
        response.update(success=True, forecast=forecast, message="Forecast generated successfully.")
    except Exception as e:
        response.update(success=False, forecast={}, message=str(e))
    return response
