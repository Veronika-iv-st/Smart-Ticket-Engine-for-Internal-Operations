from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from app.core import procesar_ticket

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/enviar", response_class=HTMLResponse)
async def enviar_ticket(
    request: Request,
    mensaje: str = Form(...),
    empleado: str = Form(...)
):
    resultado, departamento = procesar_ticket(mensaje, empleado)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "mensaje_resultado": resultado,
        "departamento": departamento,
        "mensaje_original": mensaje
    })

if __name__ == "__main__":
    empleado = input("Nombre del empleado: ")
    ticket = input("Escribe tu ticket: ")
    resultado, departamento = procesar_ticket(ticket, empleado)
    print("\n✅ Resultado:", resultado)
    print("📂 Departamento:", departamento)
