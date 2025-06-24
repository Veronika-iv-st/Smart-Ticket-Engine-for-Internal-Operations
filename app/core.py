import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DEPARTAMENTOS = {
    "soporte tecnico": "soporte_tecnico.txt",
    "recursos humanos": "recursos_humanos.txt",
    "operaciones": "operaciones.txt"
}

# 🧠 Clasificador IA con funciones bien definidas
modelo_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate.from_template("""
Clasifica el siguiente mensaje interno en uno de estos tres departamentos, basándote en las funciones reales que realiza cada uno:

- soporte tecnico: problemas con ordenadores, portátiles, redes, correo electrónico, software, herramientas digitales, dispositivos tecnológicos, acceso a internet, sistemas o impresoras.

- recursos humanos: gestiona asuntos internos del personal como nóminas, vacaciones, ausencias, contratos de empleados, bajas médicas, asistencia, permisos o conflictos laborales.

- operaciones: se encarga de temas estratégicos y externos como problemas con clientes, contratos con proveedores, ventas, cierres de contratos, informes empresariales, documentación externa o relación con terceros.

Mensaje: {ticket}

Responde solo con una palabra en minúsculas:
soporte tecnico, recursos humanos u operaciones.
""")

clasificador_ia: Runnable = prompt | modelo_llm | (lambda x: x.content.strip().lower())

def clasificar_departamento(ticket: str) -> str:
    return clasificador_ia.invoke({"ticket": ticket})

def cargar_tickets_y_nombres(ruta_txt: str) -> list[tuple[list[str], str]]:
    """Devuelve una lista de tuplas: ([nombres], contenido_ticket)"""
    if not os.path.exists(ruta_txt):
        return []
    resultado = []
    with open(ruta_txt, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith("[") and "]" in linea:
                nombres, texto = linea.split("]", 1)
                nombres = [n.strip() for n in nombres[1:].split(",") if n.strip()]
                resultado.append((nombres, texto.strip()))
    return resultado

def guardar_todos_los_tickets(ruta_txt: str, lista: list[tuple[list[str], str]]):
    with open(ruta_txt, "w", encoding="utf-8") as f:
        for nombres, texto in lista:
            linea = f"[{', '.join(nombres)}] {texto}"
            f.write(linea.strip() + "\n")

def construir_faiss_index(tickets: list[str], embeddings) -> FAISS:
    docs = [Document(page_content=t) for t in tickets]
    return FAISS.from_documents(docs, embeddings)

def buscar_similar(ticket: str, faiss_index, embeddings, threshold: float = 0.90):
    if not faiss_index:
        return None
    embedding_ticket = embeddings.embed_query(ticket)
    resultados = faiss_index.similarity_search_by_vector(embedding_ticket, k=1)
    if not resultados:
        return None

    similitud = cosine_similarity(
        [embedding_ticket],
        [embeddings.embed_query(resultados[0].page_content)]
    )[0][0]

    if similitud >= threshold:
        return resultados[0].page_content
    return None

def procesar_ticket(ticket: str, empleado: str):
    embeddings = OpenAIEmbeddings()
    departamento = clasificar_departamento(ticket)
    archivo = f"data/{DEPARTAMENTOS[departamento]}"

    # Cargar tickets ya guardados
    tickets_con_nombres = cargar_tickets_y_nombres(archivo)
    solo_textos = [t[1] for t in tickets_con_nombres]

    faiss_index = construir_faiss_index(solo_textos, embeddings) if solo_textos else None
    duplicado = buscar_similar(ticket, faiss_index, embeddings)

    if duplicado:
        actualizado = []
        for nombres, texto in tickets_con_nombres:
            if texto == duplicado:
                if empleado not in nombres:
                    nombres.insert(0, empleado)
                actualizado.append((nombres, texto))
            else:
                actualizado.append((nombres, texto))
        guardar_todos_los_tickets(archivo, actualizado)
        return f"Ya existe un ticket similar:\n“{duplicado}”", departamento

    # Ticket nuevo
    tickets_con_nombres.append(([empleado], ticket))
    guardar_todos_los_tickets(archivo, tickets_con_nombres)
    return "Ticket guardado correctamente.", departamento

if __name__ == "__main__":
    empleado = input("Nombre del empleado: ")
    ticket = input("Escribe tu ticket: ")
    resultado, departamento = procesar_ticket(ticket, empleado)
    print("\n✅ Resultado:", resultado)
    print("📂 Departamento:", departamento)
