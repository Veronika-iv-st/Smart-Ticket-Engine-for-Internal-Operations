# Smart-Ticket-Engine-for-Internal-Operations
AI-driven classification, duplicate handling, and team coordination — built with FastAPI.


🧠 Overview
This project provides a minimal yet powerful internal ticketing solution powered by OpenAI's GPT-3.5. Designed for small to medium-sized businesses, the system enables employees to submit issues or requests through a user-friendly interface. The ticket is automatically classified by department using AI, and duplicate tickets are detected and consolidated.



## 🖼️ Preview

Here’s a visual preview of the Smart Ticket Engine interface in action:

![Smart Ticket Interface](media/Captura%20de%20pantalla%202025-06-24%20222440.png)



The system enhances internal communication, reduces redundant tasks, and ensures teams only work on relevant, verified incidents.

🔧 Key Features
✅ AI-Based Department Classification
When an employee submits a new ticket, GPT-3.5 evaluates the message and determines whether it belongs to:

Operations: Issues related to customers, provider contracts, sales, contract closures, or business reports.

Technical Support: Requests about malfunctions, equipment, software, or hardware.

Human Resources: Inquiries about personal documents, salaries, leaves, team conflicts, or internal HR policies.

🧠 Smart Duplicate Detection
Each new ticket is compared in real-time with past messages from the same department. If the issue has already been reported, the system:

Notifies that the ticket is a duplicate.

Appends the reporting employee's name to the existing ticket, helping track how many people are affected by the same issue.

📝 File-Based Storage for Simplicity
Instead of using a database, tickets are saved in .txt files per department. This makes the system lightweight, transparent, and easy to maintain:

Departments can view and edit ticket files directly.

Completed tasks are deleted manually from the .txt file.

Every time a new ticket is processed, the system re-indexes the content for updated duplicate detection.

🌐 Clean Web Interface
The app offers a simple interface with:

Text input for ticket content

Field to enter the employee name

Visual confirmation of ticket processing and classification

⚙️ Installation & Setup

📁 1. Clone the Repository
git clone https://github.com/yourusername/internal-ticket-ai.git
cd internal-ticket-ai

🐍 2. Create and Activate a Virtual Environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux

📦 3. Install Required Dependencies
pip install fastapi uvicorn langchain langchain-openai langchain-community scikit-learn python-dotenv jinja2
Make sure you're using LangChain 0.3+ and GPT 3.5 or later via langchain-openai.

🔑 4. Set Your OpenAI API Key
Create a .env file in the root folder with the following content:

env
OPENAI_API_KEY=your_openai_api_key_here


📁 **Project Structure**

```
.
├── app/
│   ├── main.py                 # FastAPI app and endpoints
│   ├── core.py                 # Ticket processing and classification logic
│   └── templates/
│       └── index.html          # Frontend interface
├── data/                       # Ticket files per department (editable manually)
│   ├── operaciones.txt
│   ├── soporte_tecnico.txt
│   └── recursos_humanos.txt
├── media/                      # Screenshots or media (optional)
│   └── Captura de pantalla 2025-06-24 222440.png
├── .env                        # Contains your OpenAI API key
├── requirements.txt            # Optional - for saving dependencies
└── README.md
```



🔄 How It Works (Step-by-Step)
The employee submits a ticket using the web form with two inputs:

Their name

The content of the request or issue

The system sends the message to GPT-3.5 via LangChain using a carefully crafted prompt that defines the responsibilities of each department:

🧾 Operaciones: clientes, contratos con proveedores, ventas, informes, cierres

🛠 Soporte técnico: impresoras, ordenadores, software, averías

👥 Recursos humanos: nóminas, vacaciones, conflictos internos, documentación personal

GPT classifies the ticket into the most appropriate department.

The system reads the .txt file corresponding to the department and:

Searches for semantic duplicates using FAISS + cosine similarity.

If a match is found, it appends the new employee's name next to the existing line.

If no match is found, it saves a new line with the ticket and the employee's name.

Each time a ticket is submitted, the vector index is regenerated live — ensuring that duplicate detection stays updated without needing persistent storage.

🏢 Adaptability to Any Company
This system was designed for flexibility and minimal tech infrastructure:

🔁 No database required: all logic is based on .txt files, perfect for small internal teams.

💼 Easy to scale: just update the classification prompt or department logic.

🧩 Adaptable to other sectors: you can define your own department categories and customize the logic to your company’s needs.

✅ Manual validation: departments have full control over tasks; they can mark as done simply by deleting a line from the text file.


🌟 Key Benefits
✅ Zero database overhead – Ideal for lightweight internal use

🧠 AI-powered triage – Classify tickets accurately with GPT-3.5

📌 Live duplicate tracking – Consolidate identical reports across multiple employees

🧑‍🤝‍🧑 Team transparency – See how many people report the same issue

🔧 Manual resolution workflow – Just delete the ticket line once resolved

🌐 Runs locally – No external tools or SaaS platforms required

🚀 Future Projections
While this tool already adds intelligence to basic ticket flows, it lays the groundwork for a more advanced internal task system. Some future ideas include:

🔄 Admin dashboard to view, manage and filter tickets

🗃️ Database integration for scalable recordkeeping and stats

⏳ Task prioritization by urgency or recurrence

👤 Automatic assignment of tickets to the right person or team

📊 Reporting module: see most common issues, unresolved tasks, etc.

🧩 Modular prompt interface: dynamically adapt classification behavior
