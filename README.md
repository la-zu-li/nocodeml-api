# nocodeml-api

Python API for no-code machine learning tool

## How to run the API (dev environment)

To run this code, it is highly recommended to create a python virtual environment.

### 1. Creating a virtual environment

Run the following command, at the project root folder:

```bash
python -m venv .venv
```

### 2. Activating the virtual environment

if on Linux/Mac, run:

```
source .venv/bin/activate
```

if on Windows, run:

```powershell
# In cmd.exe
.venv\Scripts\activate.bat
# In PowerShell
.venv\Scripts\Activate.ps1
```

### 3. Installing dependencies

To install the dependencies, use `pip`. While staying at the root folder of the project, run:

```bash
pip install -r requirements.txt
```

### 4. Fill `.env` variables

At the root folder, create a text file named just `.env`. Copy the content of `.env.example` to this new file.

### 5. Running the API

While staying at the root folder, run the `main.py` file as such:

```bash
python main.py
```

### 6. Check if the API is running

Go to your browser. Enter the URL: `http://localhost:8000`

If a message like "Server up and running!" appears, the API is running.

### 7. Check API documentation

With the server running, to check the documentation for the API in the OpenAPI (Swagger) format, go to the URL: `http://localhost:8000/docs`

<img width="1438" height="899" alt="image" src="https://github.com/user-attachments/assets/f20784f7-414d-40ae-aa01-bbb77e77465b" />


TO check the same documentation in a different format, visit `http://localhost:8000/redoc`

<img width="1428" height="899" alt="image" src="https://github.com/user-attachments/assets/bf59db01-37ef-44d1-8ace-f16d7e6f7e25" />
