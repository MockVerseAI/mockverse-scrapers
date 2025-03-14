# Project Name

## Overview

Provide a brief description of your project. Explain what it does, its main features, and its purpose.

## Features

- List the key features of your project.
- Highlight any unique or important aspects.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MockVerseAI/mockverse-scrapers.git
   cd your-repo-name
   ```

2. **Set Up the Environment**:
   - Ensure you have Python installed.
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

3. **Environment Variables**:
   - Create a `.env` file in the root directory and add your environment variables:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

### Running the Script

To run the script directly and perform company research:

```bash
python src/company_research/company_research.py
```

### Using the API

1. **Start the FastAPI Server**:
   ```bash
   uvicorn src.main:app --reload
   ```

2. **Trigger Research via API**:
   - Send a POST request to the `/research` endpoint with a JSON body containing the `company_name`.

   Example using `curl`:
   ```bash
   curl -X POST "http://127.0.0.1:8000/research" -H "Content-Type: application/json" -d '{"company_name": "Example Company"}'
   ```

## Directory Structure

- `app.py`: Main application file containing the logic for data gathering and analysis.
- `requirements.txt`: List of dependencies required to run the project.
- `results/`: Directory where research results are saved.

## Contributing

If you would like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.